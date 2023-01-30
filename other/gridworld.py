import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt


class GridworldEnv(gym.Env):
    """
    This is a gym environment with discrete action in a grid worl
    """

    def __init__(self, grid_size=10, start_state=(0, 0), goal_state=(9, 9)):
        self.grid_size = grid_size
        self.start_state = start_state
        self.goal_state = goal_state

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.grid_size**2)

        self.world = np.zeros((self.grid_size, self.grid_size))
        self.agent_pos = self.start_state
        self.reset()

    def reset(self):
        self.world = np.zeros((self.grid_size, self.grid_size))
        self.world[self.agent_pos[0]][self.agent_pos[1]] = 1
        return self.world

    def step(self, action):

        # boundary check
        self.state = (
            max(0, min(self.state[0], self.grid_size - 1)),
            max(0, min(self.state[1], self.grid_size - 1)),
        )
        #

    def render(self, mode="human"):
        plt.imshow(self.world, cmap="hot", interpolation="nearest")
        plt.show()
