import numpy as np
from .rewards_base import RewardShaper

class DistanceBasedRewardShaper(RewardShaper):
    """
    A reward shaper that gives rewards based on the distance to the goal.
    """

    def compute_reward(self):
        """
        Compute the reward based on the distance to the goal.

        Args:
            actions (list): The actions taken by the agents.
            observations (dict): The observations returned by the environment.

        Returns:
            float: The computed reward.
        """
        # Example: reward is the negative of the distance to the goal
        positions = self.env.getPositions()
        goal = self.env.goal

        # Calculate the distance from each agent to the goal
        distances = np.sqrt(np.sum((positions - goal) ** 2, axis=1))

        # The reward could be negative of the sum of distances
        reward = -np.sum(distances)
        return reward
