from copy import copy
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.special import softmax
import gym
from gym import spaces
from matplotlib import cm, colors

class GoalWrapper:
    def __init__(self, env, trayectories):
        self.trayectories = trayectories
        self.env = env
    
    def step(self, actions, emb):
        obs, _, terminated, info = self.env.step(actions, emb)



class GraphEnv(gym.Env):
    def __init__(self, config,  goal, max_time=23, board_size=10, sensing_range=6, pad=3, starting_positions=None, obstacles=None):
        super(GraphEnv, self).__init__()
        """
        :starting_positions: np.array-> [nb_agents, positions]; positions == [X,Y]
                            [[0,0],
                             [1,1]]
        """
        self.config = config
        self.max_time = self.config["max_time"]
        self.board_size = self.config["board_size"][0]
        self.obstacles = self.config["obstacles"]
        self.goal = goal
        self.board =  np.zeros((self.board_size, self.board_size))
        self.pad = pad
        self.starting_positions = starting_positions 
        self.action_list = {
            1:(1,0), # Right
            2:(0,1), # Up
            3:(-1,0),# Left
            4:(0,-1), # Down
            0:(0,0)  # Idle
        }
        nb_agents = self.config["num_agents"]
        self.positionX = np.zeros((nb_agents, 1), dtype=np.int32)
        self.positionY = np.zeros((nb_agents, 1), dtype=np.int32)
        self.nb_agents = nb_agents
        self.sensing_range = sensing_range
        self.obs_shape = self.nb_agents * 4
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-15, high=15, shape=((self.obs_shape,)), dtype=np.float32
        )
        self.headings = None
        self.embedding = np.ones(self.nb_agents)
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True)
        self.mapper  =  cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        self.time = 0
        _ = self.reset()

    def reset(self):
        self.avilable_pos = np.arange(self.board_size)
        if self.starting_positions is not None:
            assert self.starting_positions.shape[0] == self.nb_agents, f"Agents and positions are not equal"
            self.positionX = self.starting_positions[:,0]
            self.positionY = self.starting_positions[:,1]
        else:
            self.positionX = np.random.choice(self.avilable_pos, size=(self.nb_agents))
            self.positionY = np.random.choice(self.avilable_pos, size=(self.nb_agents))

        # self.positionX = np.random.uniform(-2.0, 4.0, size=(self.nb_agents))
        # self.positionY = np.random.uniform(-2.0, 4.0, size=(self.nb_agents))
        self.headings = np.random.uniform(-3.14, 3.14, size=(self.nb_agents))
        self.embedding = np.ones(self.nb_agents).reshape((self.nb_agents,1))
        self.reached_goal = np.zeros(self.nb_agents)
        self._computeDistance()
        return self.getObservations()

    def _computeDistance(self):
        # Create Matrices from positions and heading
        X1, XT = np.meshgrid(self.positionX, self.positionX)
        Y1, YT = np.meshgrid(self.positionY, self.positionY)

        # Calculate distance matrix
        D_ij_x = X1 - XT
        D_ij_y = Y1 - YT
        D_ij = np.sqrt(np.multiply(D_ij_x, D_ij_x) + np.multiply(D_ij_y, D_ij_y))
        D_ij[D_ij >= self.sensing_range] = 0
      
        self.distance_matrix = D_ij
        # Get only closest 4
        self.adj_matrix = self._computeClosest(D_ij)
        self.adj_matrix[self.adj_matrix != 0] = 1

    def computeMetrics(self):
        last_state = np.array([self.positionX, self.positionY]).T

        success = np.where(last_state == self.goal)
        success_rate = len(success[0]) / self.nb_agents

        flow_time = self.computeFlowTime()

        return success_rate, flow_time

    def checkAllInGoal(self):
        last_state = np.array([self.positionX, self.positionY]).T
        success = np.where(last_state == self.goal)
        return len(success[0]) == self.nb_agents

    def check_goals(self):
        self.positionX = np.where(self.positionX_temp == self.goal[0],self.goal[0], self.positionX)
        self.positiony = np.where(self.positionY_temp == self.goal[1],self.goal[1], self.positionY)


    def computeFlowTime(self):
        if self.checkAllInGoal():
            return self.time
        else:
            return self.nb_agents * self.max_time


    @staticmethod
    def _computeClosest(A):
        for i in range(len(A)):
          temp = np.sort(A[i][A[i] != 0])
          if len(temp) < 4:
            temp = np.concatenate((np.zeros(4-len(temp)), temp))
          A[i][A[i] > temp[3]] = 0
        return A

    def step(self, actions, emb):
        """
        Actions: {
          vx[list], shape(nb_agents)
          vy[list], shape(nb_agents)
        }
        """
        done = False
        self._updateEmbedding(emb)
        self._updatePositions(actions)
        self._computeDistance()
        obs = self.getObservations()
        if self.checkAllInGoal():
          done = True
        return obs, {}, done, {}

    def _updatePositions(self, actions):

        action_x = np.array([self.action_list[act][0] for act in actions])        
        action_y = np.array([self.action_list[act][1] for act in actions])        
        self.positionX_temp = copy(self.positionX)
        self.positionY_temp = copy(self.positionY)
        self.positionX += action_x
        self.positionY += action_y
        self.check_goals()
        self.check_boundary()
        self.check_collisions()
        self.updateBoard()
        # self.positionX += actions["vx"]
        # self.positionY += actions["vy"]
        # self.headings  += actions["headings"]

    def _updateEmbedding(self, H):
      self.embedding = H
    
    def map_goal(self, agent):
            
        if self.goal[agent][0] <= self.posx[agent]+3 or self.goal[agent][0] >= self.posx[agent]-2:
            goal_x = self.goal[agent][0]
        elif self.goal[agent][0] <= self.posx[agent]+3:
            goal_x = self.goal[agent][0] - 2
        elif self.goal[agent][0] >= self.posx[agent]-2:
            goal_x = self.goal[agent][0] + 3

        if self.goal[agent][1] <= self.posy[agent]+3 or self.goal[agent][1] >= self.posy[agent]-2:
            goal_y = self.goal[agent][1]
        elif self.goal[agent][1] <= self.posy[agent]+3:
            goal_y = self.goal[agent][1] - 2
        elif self.goal[agent][1] >= self.posy[agent]-2:
            goal_y = self.goal[agent][0] + 3
        goal = np.clip(np.array([goal_x, goal_y]),0,5)
        return goal[0], goal[1]
                

    def preprocessObs(self):
        self.posx = self.positionX + self.pad
        self.posy = self.positionY + self.pad
        map_padded = np.pad(self.board,(self.pad,self.pad))
        FOV = np.zeros((self.nb_agents, 2, (self.pad*2)-1,(self.pad*2)-1))
        for agent in range(self.nb_agents):
            FOV[agent, 0, :, :] = map_padded[self.posx[agent]-2:self.posx[agent]+3,self.posy[agent]-2:self.posy[agent]+3]
            gx, gy = self.map_goal(agent=agent)
            FOV[agent, 1, gx-3, gy-3] = 3
        return FOV


    def getObservations(self):
        obs = {
            # "positionX": self.positionX,
            # "positionY": self.positionY,
            # "headings": self.headings,
            "board":self.board,
            "fov":self.preprocessObs(),
            "adj_matrix": self.adj_matrix,
            "distances": self.distance_matrix, 
            "embeddings":self.embedding
        }
        return obs

    def render(self, agentId=0, printNeigh=False, mode="plot"):
        
        plt.axis('off')
        plt.scatter(self.goal[:, 0], self.goal[:, 1], color="blue", marker="*")
        if agentId is not None:
            column = np.where(self.adj_matrix[agentId])
            column = column[0]
            for i in range(len(column)):
                plt.plot(
                    [self.positionX[agentId], self.positionX[column[i]]],
                    [self.positionY[agentId], self.positionY[column[i]]],
                    color="black",
                )
                if printNeigh:
                    neig_column = np.where(self.adj_matrix[column[i]])
                    neig_column = neig_column[0]
                    for j in range(len(neig_column)):
                        plt.plot(
                            [self.positionX[column[i]], self.positionX[neig_column[j]]],
                            [self.positionY[column[i]], self.positionY[neig_column[j]]],
                            color="black",ls="--"
                        )
        else:
            for agent in range(self.nb_agents):
                column = np.where(self.adj_matrix[agent])
                column = column[0]
                for i in range(len(column)):
                    plt.plot(
                        [self.positionX[agent], self.positionX[column[i]]],
                        [self.positionY[agent], self.positionY[column[i]]],
                        color="red",
                    )
                    if printNeigh:
                        neig_column = np.where(self.adj_matrix[column[i]])
                        neig_column = neig_column[0]
                        for j in range(len(neig_column)):
                            plt.plot(
                                [self.positionX[column[i]], self.positionX[neig_column[j]]],
                                [self.positionY[column[i]], self.positionY[neig_column[j]]],
                                color="black",
                            )

        if mode == "plot":   
            plt.scatter(self.positionX, self.positionY, s=150, color=self.mapper.to_rgba(self.embedding))
        if mode == "photo":
            plt.imshow(self.board, cmap="Greys")

        # printing FOV
        # plt.plot([self.positionX[agentId]-2, self.positionX[agentId]+2], [self.positionY[agentId]-2,self.positionY[agentId]-2], color='red')
        # plt.plot([self.positionX[agentId]-2, self.positionX[agentId]-2], [self.positionY[agentId]-2,self.positionY[agentId]+2], color='red')
        # plt.plot([self.positionX[agentId]-2, self.positionX[agentId]+2], [self.positionY[agentId]+2,self.positionY[agentId]+2], color='red')
        # plt.plot([self.positionX[agentId]+2, self.positionX[agentId]+2], [self.positionY[agentId]-2,self.positionY[agentId]+2], color='red')
        plt.plot([self.positionX[agentId]-self.sensing_range*3/4, self.positionX[agentId]+self.sensing_range*3/4], [self.positionY[agentId]-self.sensing_range*3/4,self.positionY[agentId]-self.sensing_range*3/4], color='red')
        plt.plot([self.positionX[agentId]-self.sensing_range*3/4, self.positionX[agentId]-self.sensing_range*3/4], [self.positionY[agentId]-self.sensing_range*3/4,self.positionY[agentId]+self.sensing_range*3/4], color='red')
        plt.plot([self.positionX[agentId]-self.sensing_range*3/4, self.positionX[agentId]+self.sensing_range*3/4], [self.positionY[agentId]+self.sensing_range*3/4,self.positionY[agentId]+self.sensing_range*3/4], color='red')
        plt.plot([self.positionX[agentId]+self.sensing_range*3/4, self.positionX[agentId]+self.sensing_range*3/4], [self.positionY[agentId]-self.sensing_range*3/4,self.positionY[agentId]+self.sensing_range*3/4], color='red')
        # Printing env stuff
        plt.axis([-2, self.board_size+5, -2, self.board_size+5])
        plt.plot([-1,self.board_size], [-1,-1,], color="black")
        plt.plot([-1,-1,],[self.board_size, -1], color="black")
        plt.plot([-1,self.board_size],[self.board_size, self.board_size], color="black")
        plt.plot([self.board_size, self.board_size],[self.board_size,-1], color="black")
        plt.pause(0.1)
        plt.clf()
        plt.axis('off')

    def getGraph(self):
        return self.adj_matrix

    def getEmbedding(self):
        return copy(self.embedding)

    def getPositions(self):
        return self.positionX, self.positionY
    
    def check_boundary(self):
        self.positionX[self.positionX > self.board_size - 1] = self.board_size -1
        self.positionY[self.positionY > self.board_size - 1] = self.board_size -1
        self.positionX[self.positionX < 0] = 0
        self.positionY[self.positionY < 0] = 0
    
    def updateBoard(self):
        self.board[self.positionX_temp, self.positionY_temp] = 0
        self.board[self.positionX, self.positionY] = 1
        
    def check_collisions(self):
        """
        Iterate over X:
        if 2 equal, check their Y:
            If equal, revert to position in temp
        """
        ck = {}
        for i in range(len(self.positionX)):
            hash = str((self.positionX[i],self.positionY[i]))
            if hash in ck:
                self.positionX[i] = self.positionX_temp[i]
                self.positionY[i] = self.positionY_temp[i]
                self.positionX[int(ck[hash])] = self.positionX_temp[int(ck[hash])]
                self.positionY[int(ck[hash])] = self.positionY_temp[int(ck[hash])]
                continue
            ck[hash] = i

    def printBoard(self):
        self.updateBoard()
        return f"Game Board:\n{self.board}"

########## utils ##########
def create_goals(board_size, num_agents):
    avilable_pos = np.arange(board_size[0])
    goals_x = np.random.choice(avilable_pos, size=num_agents,replace=False)
    goals_y = np.random.choice(avilable_pos, size=num_agents,replace=False)
    goals = np.array([goals_x, goals_y]).T
    return goals


if __name__ == "__main__":
    agents = 3
    board_size=16
    sensing = 4
    goals = np.array([[3,4],[6,2],[9,1]])
    env = GraphEnv(agents, goal=goals, board_size=board_size, sensing_range=sensing)
    emb = np.ones(agents).reshape((agents,1))
    obs = env.reset()
    for i in range(100):
        actions = np.random.randint(0,4,size=(agents))
        obs, _, _, _ = env.step(actions,emb)
        env.render(agentId=0, printNeigh=True)
        break
    
    pprint(obs)
