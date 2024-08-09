import yaml
import numpy as np
from copy import copy
import gymnasium as gym
from gymnasium import spaces
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from rewards.rewards_distance import DistanceBasedRewardShaper

class GoalWrapper:
    def __init__(self, env, trayectories):
        self.trayectories = trayectories
        self.env = env

    def step(self, actions, emb):
        obs, _, terminated, info = self.env.step(actions, emb)



class GraphEnv(gym.Env):
    """
    A custom environment for a grid-based multi-agent navigation task.

    Attributes:
        config (dict): Configuration dictionary containing environment parameters.
        goal (np.array): Array containing goal positions for the agents.
        starting_positions (np.array, optional): Array of starting positions for each agent.
        obstacles (np.array, optional): Array of obstacle positions.
        action_list (dict): Dictionary mapping action indices to movement directions.
        action_space (gymnasium.spaces.Discrete): Discrete action space for the agents.
        observation_space (gymnasium.spaces.Box): Continuous observation space for the environment.
        time (int): Current time step in the environment.
    """

    def __init__(
        self,
        config,
        goal,
        starting_positions=None,
        obstacles=None,
    ):
        super(GraphEnv, self).__init__()
        
        self.config = config
        self.unpack_config()
        self.goal = goal
        
        self.obstacles = obstacles if obstacles is not None else None
        self.starting_positions = starting_positions

                # Initialize the reward shaper based on config
        reward_shaper_class = config.get("reward_shaper_class", DistanceBasedRewardShaper)
        self.reward_shaper = reward_shaper_class(config, self)

        # Define the action list for the agents
        self.action_list = {
            1: (1, 0),  # Right
            2: (0, 1),  # Up
            3: (-1, 0),  # Left
            4: (0, -1),  # Down
            0: (0, 0),  # Idle
        }
        
        # Define the action and observation spaces
        self.action_space = spaces.Discrete(5)
        # Define the observation space based on the different components of observations
        self.observation_space = spaces.Dict({
            "board": spaces.Box(
                low=0.0, high=4.0, shape=(self.board_size, self.board_size), dtype=np.float32
            ),
            "fov": spaces.Box(
                low=0.0, high=3.0, shape=(self.num_agents, 2, self.pad*2 - 1, self.pad*2 - 1), dtype=np.float32
            ),
            "adj_matrix": spaces.Box(
                low=0.0, high=1.0, shape=(self.num_agents, self.num_agents), dtype=np.float32
            ),
            "distances": spaces.Box(
                low=0.0, high=self.sensing_range, shape=(self.num_agents, self.num_agents), dtype=np.float32
            ),
            "embeddings": spaces.Box(
                low=0.0, high=1.0, shape=(self.num_agents, 1), dtype=np.float32
            ),
        })
        
        # Initialize the color mapper for visualization
        norm = colors.Normalize(vmin=0.0, vmax=1.4, clip=True)
        self.mapper = cm.ScalarMappable(norm=norm, cmap=cm.inferno)
        
        # Initialize time and reset the environment
        self.time = 0
        _ = self.reset()

    def unpack_config(self):
        """
        Unpack the configuration dictionary and initialize environment parameters.
        """
        self.max_time = self.config["max_time"]
        self.min_time = self.config["min_time"]
        self.pad = self.config["pad"]
        self.board_size = self.config["board_size"][0]
        self.board = np.zeros((self.board_size, self.board_size))
        self.sensing_range = self.config["sensing_range"]
        self.num_agents = self.config["num_agents"]
        self.positionX = np.zeros((self.num_agents, 1), dtype=np.int32)
        self.positionY = np.zeros((self.num_agents, 1), dtype=np.int32)
        self.obs_shape = self.num_agents * 4
        self.embedding = np.ones(self.num_agents)
        self.headings = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Returns:
            dict: Initial observation of the environment.
        """
        self.time = 0
        self.avilable_pos = np.arange(self.board_size)

        # Place obstacles on the board if provided
        if self.obstacles is not None:
            self.board[self.obstacles[:, 1], self.obstacles[:, 0]] = 2

        # Set agent starting positions
        if self.starting_positions is not None:
            assert (
                self.starting_positions.shape[0] == self.num_agents
            ), "Number of agents and starting positions do not match."
            self.positionX = self.starting_positions[:, 0]
            self.positionY = self.starting_positions[:, 1]
        else:
            # Randomly initialize agent positions if not provided
            self.avilable_pos_x = np.arange(self.board_size)
            self.avilable_pos_y = np.arange(self.board_size)
            if self.obstacles is not None:
                mask_x = np.isin(self.avilable_pos_x, self.obstacles[:, 0])
                mask_y = np.isin(self.avilable_pos_y, self.obstacles[:, 1])
                self.avilable_pos_x = self.avilable_pos_x[~mask_x]
                self.avilable_pos_y = self.avilable_pos_y[~mask_y]
            self.positionX = np.random.choice(
                self.avilable_pos_x, size=(self.num_agents)
            )
            self.positionY = np.random.choice(
                self.avilable_pos_y, size=(self.num_agents)
            )

        self.goal_paded = self.goal + self.pad
        self.headings = np.random.uniform(-3.14, 3.14, size=(self.num_agents))
        self.embedding = np.ones(self.num_agents).reshape((self.num_agents, 1))
        self.reached_goal = np.zeros(self.num_agents)
        self._computeDistance()
        return self.getObservations(), {}

    def step(self, actions, emb):
        """
        Execute a step in the environment.

        Args:
            actions (list): List of actions for each agent.
            emb (np.array): Embedding matrix.

        Returns:
            tuple: (observations, reward, done, info_dict)
        """
        done = False
        self._updateEmbedding(emb)
        self._updatePositions(actions)
        self._computeDistance()
        obs = self.getObservations()
        self.time += 1
        if self.checkAllInGoal():
            done = True
        # Compute the reward using the reward shaper
        reward = self.reward_shaper.compute_reward()
        
        return obs, reward, done, {}

    def getObservations(self):
        """
        Get the current observations of the environment.

        Returns:
            dict: Observations including the board, field of view, adjacency matrix, 
                  distances, and embeddings.
        """
        obs = {
            "board": self.updateBoardGoal(),
            "fov": self.preprocessObs(),
            "adj_matrix": self.adj_matrix,
            "distances": self.distance_matrix,
            "embeddings": self.embedding,
        }
        return obs

    def getGraph(self):
        """
        Get the current adjacency matrix of the environment.

        Returns:
            np.array: Adjacency matrix representing agent connections.
        """
        return self.adj_matrix

    def getEmbedding(self):
        """
        Get the current embedding of the agents.

        Returns:
            np.array: Embedding matrix.
        """
        return copy(self.embedding)

    def getPositions(self):
        """
        Get the current positions of the agents.

        Returns:
            np.array: Array of agent positions [X, Y].
        """
        return np.array([self.positionX, self.positionY]).T

    def _computeDistance(self):
        """
        Compute the distance matrix and adjacency matrix for the agents.
        """
        # Create matrices from positions and heading
        X1, XT = np.meshgrid(self.positionX, self.positionX)
        Y1, YT = np.meshgrid(self.positionY, self.positionY)

        # Calculate distance matrix
        D_ij_x = X1 - XT
        D_ij_y = Y1 - YT
        D_ij = np.sqrt(np.multiply(D_ij_x, D_ij_x) + np.multiply(D_ij_y, D_ij_y))
        D_ij[D_ij >= self.sensing_range] = 0

        self.distance_matrix = D_ij
        # Get only the closest 4 connections
        self.adj_matrix = self._computeClosest(D_ij)
        self.adj_matrix[self.adj_matrix != 0] = 1

    def computeMetrics(self):
        """
        Compute performance metrics for the environment.

        Returns:
            tuple: success_rate (float), flow_time (int).
        """
        last_state = np.array([self.positionX, self.positionY]).T
        success = last_state[last_state == self.goal]
        success_rate = len(success) / 2
        flow_time = self.computeFlowTime()
        return success_rate, flow_time

    def checkAllInGoal(self):
        """
        Check if all agents have reached their goal positions.

        Returns:
            bool: True if all agents are in goal positions, False otherwise.
        """
        last_state = np.array([self.positionX, self.positionY]).T
        return np.array_equal(last_state, self.goal)

    def check_goals(self):
        """
        Update agent positions if they have reached their goals.
        """
        positions = np.array([self.positionX, self.positionY]).T
        positions = np.where(positions == self.goal, self.goal, positions)
        self.positionX, self.positionY = positions[:, 0], positions[:, 1]

    def computeFlowTime(self):
        """
        Compute the total flow time for the agents to reach their goals.

        Returns:
            int: Flow time.
        """
        if self.checkAllInGoal():
            return self.time
        else:
            return self.num_agents * self.max_time

    @staticmethod
    def _computeClosest(A):
        """
        Compute the closest 4 connections for each agent based on the distance matrix.

        Args:
            A (np.array): Distance matrix.

        Returns:
            np.array: Modified distance matrix with only the closest 4 connections.
        """
        for i in range(len(A)):
            temp = np.sort(A[i][A[i] != 0])
            if len(temp) < 4:
                temp = np.concatenate((np.zeros(4 - len(temp)), temp))
            A[i][A[i] > temp[3]] = 0
        return A


    def _updatePositions(self, actions):
        """
        Update the positions of the agents based on their actions.
        """
        action_x = np.array([self.action_list[act][0] for act in actions])
        action_y = np.array([self.action_list[act][1] for act in actions])
        self.positionX_temp = copy(self.positionX)
        self.positionY_temp = copy(self.positionY)
        self.positionX += action_x
        self.positionY += action_y
        self.check_goals()
        if self.obstacles is not None:
            self.check_collision_obstacle()
        self.check_boundary()
        self.check_collisions()
        self.updateBoard()

    def _updateEmbedding(self, H):
        """
        Update the embedding matrix for the agents.

        Args:
            H (np.array): New embedding matrix.
        """
        self.embedding = H

    def map_goal(self, agent):
        """
        Map the goal to the agent's field of view (FOV).

        Args:
            agent (int): Index of the agent.

        Returns:
            tuple: (goal_y, goal_x) mapped goal position in the agent's FOV.
        """
        # Check if the goal is within the agent's FOV
        if (
            self.goal_paded[agent][0] < self.posx[agent] + self.pad - 1
            and self.goal_paded[agent][0] > self.posx[agent] - self.pad + 1
        ):
            goal_x = -(self.posx[agent] - self.goal_paded[agent][0]) + self.pad - 1

        elif self.goal_paded[agent][0] <= self.posx[agent] - self.pad + 1:
            goal_x = 0
        elif self.goal_paded[agent][0] >= self.posx[agent] + self.pad - 1:
            goal_x = 1 + self.pad

        # Same for Y
        if (
            self.goal_paded[agent][1] < self.posy[agent] + self.pad - 1
            and self.goal_paded[agent][1] >= self.posy[agent] - self.pad + 1
        ):
            goal_y = (self.posy[agent] - self.goal_paded[agent][1]) + self.pad - 1

        elif self.goal_paded[agent][1] <= self.posy[agent] - self.pad + 1:
            goal_y = 1 + self.pad

        elif self.goal_paded[agent][1] >= self.posy[agent] + self.pad - 1:
            goal_y = 0

        goal = np.array([goal_y, goal_x])  # Reversed for numpy
        return goal[0], goal[1]

    def preprocessObs(self):
        """
        Preprocess the observations for the agents' fields of view (FOV).

        Returns:
            np.array: Preprocessed FOV for each agent.
        """
        self.posx = self.positionX + self.pad
        self.posy = self.positionY + self.pad
        map_padded = np.pad(self.board, (self.pad, self.pad))
        FOV = np.zeros((self.num_agents, 2, (self.pad * 2) - 1, (self.pad * 2) - 1))

        for agent in range(self.num_agents):
            FOV[agent, 0, :, :] = np.flip(
                map_padded[
                    self.positionY[agent] + 1 : self.positionY[agent] + 6,
                    self.positionX[agent] + 1 : self.positionX[agent] + 6,
                ],
                axis=0,
            )
            gx, gy = self.map_goal(agent=agent)
            FOV[agent, 1, gx, gy] = 3

        return FOV

    def check_boundary(self):
        """
        Ensure agent positions stay within the boundaries of the board.
        """
        self.positionX[self.positionX > self.board_size - 1] = self.board_size - 1
        self.positionY[self.positionY > self.board_size - 1] = self.board_size - 1
        self.positionX[self.positionX < 0] = 0
        self.positionY[self.positionY < 0] = 0

    def updateBoard(self):
        """
        Update the game board with the agents' current positions.
        """
        self.board[self.positionY_temp, self.positionX_temp] = 0
        self.board[self.positionY, self.positionX] = 1

    def updateBoardGoal(self):
        """
        Update the game board with the goal positions.
        
        Returns:
            np.array: Updated board with goal positions.
        """
        board = copy(self.board)
        board[self.goal[:, 1], self.goal[:, 0]] += 4
        return board

    def check_collisions(self):
        """
        Check for collisions between agents and revert their positions if necessary.
        """
        ck = {}
        for i in range(len(self.positionX)):
            hash = str((self.positionX[i], self.positionY[i]))
            if hash in ck:
                self.positionX[i] = self.positionX_temp[i]
                self.positionY[i] = self.positionY_temp[i]
                self.positionX[int(ck[hash])] = self.positionX_temp[int(ck[hash])]
                self.positionY[int(ck[hash])] = self.positionY_temp[int(ck[hash])]
                continue
            ck[hash] = i

    def check_collision_obstacle(self):
        """
        Check for collisions between agents and obstacles, and revert their positions if necessary.
        """
        ck = {
            str((self.obstacles[i][0], self.obstacles[i][1])): i
            for i in range(len(self.obstacles))
        }
        for i in range(len(self.positionX)):
            hash = str((self.positionX[i], self.positionY[i]))
            if hash in ck:
                self.positionX[i] = self.positionX_temp[i]
                self.positionY[i] = self.positionY_temp[i]

    def printBoard(self):
        """
        Print the current state of the game board.

        Returns:
            str: Formatted string representing the game board.
        """
        self.updateBoard()
        return f"Game Board:\n{self.board}"


    def render(self, agentId=0, printNeigh=False, printFOV=False, mode="plot"):

        plt.axis("off")
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
            for agent in range(self.num_agents):
                column = np.where(self.adj_matrix[agent])
                column = column[0]
                for i in range(len(column)):
                    plt.plot(
                        [self.positionX[agent], self.positionX[column[i]]],
                        [self.positionY[agent], self.positionY[column[i]]],
                        color="black",
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

        if mode == "plot":
            plt.scatter(
                self.positionX,
                self.positionY,
                s=150,
                color=self.mapper.to_rgba(self.embedding),
            )
            plt.scatter(
                self.goal[:, 0], self.goal[:, 1], color="blue", marker="*", s=150
            )
            if self.obstacles is not None:
                plt.scatter(
                    self.obstacles[:, 0],
                    self.obstacles[:, 1],
                    color="black",
                    marker="s",
                    s=150,
                )
        if mode == "photo":
            plt.imshow(self.board, cmap="Greys")

        # printing FOV
        if printFOV:
            plt.plot(
                [
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                ],
                color="red",
            )
            plt.plot(
                [
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                ],
                color="red",
            )
            plt.plot(
                [
                    self.positionX[agentId] - self.sensing_range * 3 / 4,
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                ],
                color="red",
            )
            plt.plot(
                [
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                    self.positionX[agentId] + self.sensing_range * 3 / 4,
                ],
                [
                    self.positionY[agentId] - self.sensing_range * 3 / 4,
                    self.positionY[agentId] + self.sensing_range * 3 / 4,
                ],
                color="red",
            )
        # Printing env stuff
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
        plt.pause(0.1)
        plt.clf()
        plt.axis("off")

    def report_observations(self, observations):
        """
        Print relevant information from the observations, including shape, min, max, and mean.

        Args:
            observations (dict): A dictionary of observations returned by getObservations().
        """
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                print(f"Observation: {key}")
                print(f"  Shape: {value.shape}")
                print(f"  Min: {np.min(value)}")
                print(f"  Max: {np.max(value)}")
                print(f"  Mean: {np.mean(value)}")
                print(f"  Std: {np.std(value)}")
                print("-" * 30)
            else:
                print(f"Observation: {key}")
                print(f"  Type: {type(value)}")
                print("-" * 30)



def load_config(config_path):
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main(config_path, starting_positions, goals, obstacles, actions):
    """
    Main function to run the GraphEnv environment.

    Args:
        config_path (str): Path to the YAML configuration file.
        starting_positions (np.array): Array of starting positions for the agents.
        goals (np.array): Array of goal positions for the agents.
        obstacles (np.array): Array of obstacle positions.
        actions (np.array): Array of actions for each agent over time.
    """
    # Load configuration
    config = load_config(config_path)

    # Initialize the environment
    env = GraphEnv(
        config,
        goal=goals,
        starting_positions=starting_positions,
        obstacles=obstacles,
    )

    # Reset environment and get initial observation
    obs, _ = env.reset()

    # Placeholder for embedding; assuming a simple embedding for demonstration
    emb = np.ones(config["num_agents"]).reshape((config["num_agents"], 1))

    # Run the environment for each set of actions
    for i in range(actions.shape[0]):
        obs, rewards, done, _ = env.step(actions[i, :], emb)
        print(rewards)
        env.render(agentId=0, printNeigh=True)
        if done:
            break

    # Optionally, report observation statistics
    env.report_observations(obs)

if __name__ == "__main__":
    config_path = 'configs\config_test.yaml'
    
    # Define your initial positions, goals, obstacles, and actions
    starting_positions = np.array([[6, 6], [3, 3]])
    goals = np.array([[4, 3], [7, 7]])
    obstacles = np.array([[2, 2], [2, 3]])

    # Actions are predefined as a series of steps
    actions = np.zeros((7, 2))
    actions[:, 0] = np.array([4, 4, 4, 3, 3, 3, 3]).T
    actions[:, 1] = np.array([0, 0, 0, 3, 4, 4, 4]).T

    main(config_path, starting_positions, goals, obstacles, actions)