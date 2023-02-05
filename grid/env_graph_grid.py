import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.special import softmax
import gym
from gym import spaces
from matplotlib import cm, colors


class GraphEnv(gym.Env):
    def __init__(self, nb_agents, board_size=10, sensing_range=6):
        super(GraphEnv, self).__init__()

        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size))

        self.action_list = {
            0: (1, 0),  # Right
            1: (0, 1),  # Up
            2: (-1, 0),  # Left
            3: (0, -1),  # Down
            4: (0, 0),  # Idle
        }

        self.positionX = np.zeros((nb_agents, 1))
        self.positionY = np.zeros((nb_agents, 1))
        self.nb_agents = nb_agents
        self.sensing_range = sensing_range
        self.obs_shape = self.nb_agents * 4
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-15, high=15, shape=((self.obs_shape,)), dtype=np.float32
        )
        self.graph = nx.Graph()
        self.headings = None
        self.embedding = np.ones(self.nb_agents)
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True)
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)

        _ = self.reset()

    def reset(self):
        self.avilable_pos = np.arange(self.board_size)

        self.positionX = np.random.choice(self.avilable_pos, size=(self.nb_agents))
        self.positionY = np.random.choice(self.avilable_pos, size=(self.nb_agents))

        # self.positionX = np.random.uniform(-2.0, 4.0, size=(self.nb_agents))
        # self.positionY = np.random.uniform(-2.0, 4.0, size=(self.nb_agents))
        self.headings = np.random.uniform(-3.14, 3.14, size=(self.nb_agents))
        self.embedding = np.ones(self.nb_agents).reshape((self.nb_agents, 1))
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

    @staticmethod
    def _computeClosest(A):
        for i in range(len(A)):
            temp = np.sort(A[i][A[i] != 0])
            if len(temp) < 4:
                temp = np.concatenate((np.zeros(4 - len(temp)), temp))
            A[i][A[i] > temp[3]] = 0
        return A

    def step(self, actions, emb):
        """
        Actions: {
          vx[list], shape(nb_agents)
          vy[list], shape(nb_agents)
        }
        """
        self._updateEmbedding(emb)
        self._updatePositions(actions)
        self._computeDistance()
        obs = self.getObservations()
        return obs, {}, {}, {}

    def _updatePositions(self, actions):

        action_x = np.array([self.action_list[act][0] for act in actions])
        action_y = np.array([self.action_list[act][1] for act in actions])
        self.positionX += action_x
        self.positionY += action_y
        self.check_boundary()
        # self.positionX += actions["vx"]
        # self.positionY += actions["vy"]
        # self.headings  += actions["headings"]

    def _updateEmbedding(self, H):
        self.embedding = H

    def getObservations(self):
        obs = {
            "positionX": self.positionX,
            "positionY": self.positionY,
            # "headings": self.headings,
            "adj_matrix": self.adj_matrix,
            "distances": self.distance_matrix,
            "embeddings": self.embedding,
        }
        return obs

    def render(self, agentId=None, printNeigh=False):
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
                            color="black",
                            ls="--",
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
                                [
                                    self.positionX[column[i]],
                                    self.positionX[neig_column[j]],
                                ],
                                [
                                    self.positionY[column[i]],
                                    self.positionY[neig_column[j]],
                                ],
                                color="black",
                            )

        plt.scatter(
            self.positionX,
            self.positionY,
            s=150,
            color=self.mapper.to_rgba(self.embedding),
        )

        plt.axis([-2, self.board_size + 5, -2, self.board_size + 5])
        plt.plot(
            [-1, self.board_size],
            [
                -1,
                -1,
            ],
            color="black",
        )
        plt.plot(
            [
                -1,
                -1,
            ],
            [self.board_size, -1],
            color="black",
        )
        plt.plot(
            [-1, self.board_size], [self.board_size, self.board_size], color="black"
        )
        plt.plot(
            [self.board_size, self.board_size], [self.board_size, -1], color="black"
        )
        plt.pause(0.00001)
        plt.axis("off")
        plt.clf()
        plt.axis("off")

    def getGraph(self):
        return self.adj_matrix

    def getPositions(self):
        return self.positionX, self.positionY

    def check_boundary(self):
        self.positionX[self.positionX > self.board_size - 1] = self.board_size - 1
        self.positionY[self.positionY > self.board_size - 1] = self.board_size - 1
        self.positionX[self.positionX < 0] = 0
        self.positionY[self.positionY < 0] = 0

    def updateBoard(self):
        self.board = np.zeros((self.board_size, self.board_size))
        self.check_boundary()
        for i in range(len(self.positionX)):
            agent_pos = np.array([self.positionX[i], self.positionY[i]])
            self.board[agent_pos[0], agent_pos[1]] = 1

    def printBoard(self):
        self.updateBoard()
        return f"Game Board:\n{self.board}"


# if __name__ == "__main__":
# agents = 8
# env = GraphEnv(agents)

# obs = env.reset()
# for i in range(100):
#     actions = {
#         "vx": np.random.uniform(-1.0, 1, size=(agents)),
#         "vy": np.random.uniform(-1.0, 1, size=(agents)),
#     }
#     obs, _, _, _ = env.step(actions)
#     env.render(agentId=0)
