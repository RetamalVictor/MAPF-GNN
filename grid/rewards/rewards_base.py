from abc import ABC, abstractmethod

class RewardShaper(ABC):
    """
    Abstract base class for reward shaping.
    """

    def __init__(self, config, env):
        """
        Initialize the reward shaper.

        Args:
            config (dict): Configuration dictionary.
            env (GraphEnv): The environment instance to which this reward shaper is attached.
        """
        self.config = config
        self.env = env

    @abstractmethod
    def compute_reward(self, actions, observations):
        """
        Compute the reward based on the actions taken and the observations.

        Args:
            actions (list): The actions taken by the agents.
            observations (dict): The observations returned by the environment.

        Returns:
            float: The computed reward.
        """
        pass
