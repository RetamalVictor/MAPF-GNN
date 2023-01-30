from pprint import pprint
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from copy import copy

"""
The main engine is done, now we need to implement the control
the agents are represented in a graph like structure, where the edges are defined by the distance between the agents
"""
# class ActionDict(Enum):
#     FORWARD     = (1 , 0)
#     BACKWARDS   = (-1, 0)
#     RIGHT       = (0 , 1)
#     LEFT        = (0 ,-1)


class Agent:
    def __init__(self, id):
        self.id = str(id)
        self.neighbours = []

    def addNeighbour(self, neighbour_id):
        if str(neighbour_id) not in self.neighbours:
            self.neighbours.append(str(neighbour_id))

    def getNeighbours(self):
        return self.neighbours

    def getNbNeighbours(self):
        return len(self.neighbours)

    def removeNeighbour(self, neighbour_id):
        if str(neighbour_id) in self.neighbours:
            self.neighbours.remove(str(neighbour_id))


class DotsEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}
    """
    TODO:
    - Add assertions and error checkings with the nb_agents
    """

    def __init__(self, nb_agents, agent_random_start=False, speed=0.6):
        super(DotsEnv, self).__init__()

        self.state_space = 4 + nb_agents
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(nb_agents,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-15, high=15, shape=(nb_agents * 2,), dtype=np.float32
        )
        self.agent_random_start = agent_random_start
        self.speed = speed

        self.nb_agents = nb_agents
        self.reset()

    def step(self, action):
        # print(f"Action: {action}")
        self._updatePositions(action)
        observation = self._getObservation()
        reward = None
        done = None
        # self._changeTarget()
        return observation, reward, done, {}

    def reset(self):
        self.bound = False
        self.agents = {str(i): Agent(i) for i in range(self.nb_agents)}

        if self.agent_random_start:
            self.posX = np.array(
                [
                    np.random.randint(3, 6) * np.random.choice([-1, 1])
                    for i in range(self.nb_agents)
                ],
                dtype=np.float32,
            )
            self.posY = np.array(
                [
                    np.random.randint(3, 6) * np.random.choice([-1, 1])
                    for i in range(self.nb_agents)
                ],
                dtype=np.float32,
            )
        else:
            if self.nb_agents == 2:
                self.posX = np.array([1, -1], dtype=np.float32)
                self.posY = np.array([1, -1], dtype=np.float32)
            else:
                print("More than two agents need to be started randomly")
                self.posX = np.array(
                    [
                        np.random.randint(3, 6) * np.random.choice([-1, 1])
                        for i in range(self.nb_agents)
                    ],
                    dtype=np.float32,
                )
                self.posY = np.array(
                    [
                        np.random.randint(3, 6) * np.random.choice([-1, 1])
                        for i in range(self.nb_agents)
                    ],
                    dtype=np.float32,
                )
        distances = self._computeDistanceMatrix()
        self.populateNeighbours(distances)
        observation = self._getObservation()
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        self.colors = ["red", "green", "blue", "black", "orange", "purple", "yellow"]
        plt.scatter(self.posX, self.posY)
        plt.axis([-13, 13, -13, 13])
        plt.pause(0.00001)
        plt.clf()

    def close(self):
        ...

    def getNeigborsPositionY(self, agentId):
        """
        Returns the positions of the neighbours of the agent with the given id
        """
        neighbours = self.agents[str(agentId)].getNeighbours()
        neighbours_posY = np.array([self.posY[int(neigh)] for neigh in neighbours])
        return neighbours_posY

    def getNeigborsPositionX(self, agentId):
        neighbours = self.agents[str(agentId)].getNeighbours()
        neighbours_posX = np.array([self.posX[int(neigh)] for neigh in neighbours])
        return neighbours_posX

    def _getObservation(self):
        """
        Calculates the distance between the agents and the target.
        Agents positions are stored in two list. PosX and PosY
        """
        agent_distances = self._computeDistanceMatrix()
        self.populateNeighbours(agent_distances)
        obs = {
            str(i): {
                "own_pos": np.array([self.posX[i], self.posY[i]]),
                "neighbours_posX": self.getNeigborsPositionX(i),
                "neighbours_posY": self.getNeigborsPositionY(i),
            }
            for i in range(self.nb_agents)
        }
        return obs

    def populateNeighbours(self, distances):
        """
        distances is computed with the _computeDistanceMatrix function
        """
        for agent in range(self.nb_agents):
            for neigh in range(len(distances[agent])):
                if distances[agent][neigh] > 18.0:
                    self.agents[str(agent)].addNeighbour(neigh)
                else:
                    self.agents[str(agent)].removeNeighbour(neigh)

    def _computeDistanceMatrix(self):

        # Create Matrices from positions and heading
        X1, XT = np.meshgrid(self.posX, self.posX)
        Y1, YT = np.meshgrid(self.posY, self.posY)

        # Calculate distance matrix
        D_ij_x = X1 - XT
        D_ij_y = Y1 - YT
        D_ij = np.sqrt(np.multiply(D_ij_x, D_ij_x) + np.multiply(D_ij_y, D_ij_y))
        # D_ij[(D_ij >= 3.5) | (D_ij == 0)] = np.inf
        return D_ij

    def _preprocessObs(self):
        pass

    def check_boundary(self):
        if (
            np.any(self.posX > 12)
            or np.any(self.posX < -12)
            or np.any(self.posY > 12)
            or np.any(self.posY < -12)
        ):
            self.bound = True
            return True
        else:
            return False

    def _updatePositions(self, actions):
        """
        Actions is a dict of the form {agent_id: action}
        """
        for agent in range(len(actions)):
            self.posX[agent] += actions[str(agent)][0]
            self.posY[agent] += actions[str(agent)][1]

    def printNeighbours(self):
        for agent in self.agents:
            print(
                f"Agent {agent} has {self.agents[agent].getNbNeighbours()} neighbours"
            )

    def printDistanceMatrix(self):
        print(self._computeDistanceMatrix())

    def getAverageDistance(self):
        distance = self._computeDistanceMatrix()
        return np.mean(distance[distance != 0])


if __name__ == "__main__":
    age = 10
    env = DotsEnv(nb_agents=age)
    env.reset()
    for i in range(200):
        action = {
            str(i): [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            for i in range(age)
        }
        obs, r, dones, info = env.step(action)
        env.render()

    pprint(obs)
    env.printNeighbours()
    env.printDistanceMatrix()
    print(env.getAverageDistance())
