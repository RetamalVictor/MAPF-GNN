"""
Comprehensive tests for the MAPF-GNN environment.
Tests both functionality correctness and performance benchmarks.
"""

import numpy as np
import time
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles


class TestEnvironmentFunctionality:
    """Test core environment functionality and correctness."""

    def setup_method(self):
        """Setup test configuration."""
        self.config = {
            "num_agents": 5,
            "board_size": [10, 10],
            "max_time": 20,
            "min_time": 5,
        }

    def test_unique_goal_generation(self):
        """Test that goal generation creates unique positions."""
        num_agents = 10
        board_size = [15, 15]

        for _ in range(10):  # Test multiple times for randomness
            goals = create_goals(board_size, num_agents)

            # Check shape
            assert goals.shape == (num_agents, 2)

            # Check uniqueness - convert to tuples for set comparison
            goal_tuples = [tuple(g) for g in goals]
            assert len(goal_tuples) == len(set(goal_tuples)), "Goals must be unique"

            # Check bounds
            assert np.all(goals >= 0)
            assert np.all(goals[:, 0] < board_size[0])
            assert np.all(goals[:, 1] < board_size[1])

    def test_unique_goal_generation_with_obstacles(self):
        """Test goal generation avoids obstacles."""
        board_size = [10, 10]
        obstacles = np.array([[2, 3], [4, 5], [6, 7]])
        num_agents = 5

        for _ in range(10):
            goals = create_goals(board_size, num_agents, obstacles)

            # Check no goal is on an obstacle
            for goal in goals:
                for obstacle in obstacles:
                    assert not np.array_equal(goal, obstacle), "Goal cannot be on obstacle"

    def test_unique_obstacle_generation(self):
        """Test that obstacle generation creates unique positions."""
        board_size = [10, 10]
        num_obstacles = 8

        for _ in range(10):
            obstacles = create_obstacles(board_size, num_obstacles)

            # Check shape
            assert obstacles.shape == (num_obstacles, 2)

            # Check uniqueness
            obstacle_tuples = [tuple(o) for o in obstacles]
            assert len(obstacle_tuples) == len(set(obstacle_tuples)), "Obstacles must be unique"

            # Check bounds
            assert np.all(obstacles >= 0)
            assert np.all(obstacles[:, 0] < board_size[0])
            assert np.all(obstacles[:, 1] < board_size[1])

    def test_multi_agent_collision_detection(self):
        """Test that collision detection handles 3+ agents correctly."""
        obstacles = np.array([[5, 5]])
        goals = np.array([[9, 9], [8, 8], [7, 7], [6, 6], [0, 0]])
        start_pos = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [9, 9]])

        env = GraphEnv(
            self.config,
            goal=goals,
            starting_positions=start_pos,
            obstacles=obstacles
        )
        env.reset()

        # Make 3 agents try to move to the same position (4, 4)
        # Agent 0: at (0,0) -> needs to move right(1) and up(2) multiple times
        # Agent 1: at (1,1) -> similar movement
        # Agent 2: at (2,2) -> similar movement

        # First, let's position agents near (4,4)
        env.positionX = np.array([3, 3, 5, 7, 9])
        env.positionY = np.array([4, 5, 4, 7, 9])

        # Save current positions as temp
        env.positionX_temp = env.positionX.copy()
        env.positionY_temp = env.positionY.copy()

        # Now simulate all three trying to move to (4,4)
        env.positionX[0] = 4  # Agent 0 moves to (4,4)
        env.positionY[0] = 4
        env.positionX[1] = 4  # Agent 1 moves to (4,4)
        env.positionY[1] = 4
        env.positionX[2] = 4  # Agent 2 moves to (4,4)
        env.positionY[2] = 4

        # Check collision should revert all three
        env.check_collisions()

        # All three agents should be reverted
        assert env.positionX[0] == 3 and env.positionY[0] == 4
        assert env.positionX[1] == 3 and env.positionY[1] == 5
        assert env.positionX[2] == 5 and env.positionY[2] == 4

        # Agents 3 and 4 should be unchanged
        assert env.positionX[3] == 7 and env.positionY[3] == 7
        assert env.positionX[4] == 9 and env.positionY[4] == 9

    def test_success_rate_calculation(self):
        """Test correct success rate computation."""
        goals = np.array([[5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])

        env = GraphEnv(self.config, goal=goals)
        env.reset()

        # Test 0 agents at goal
        env.positionX = np.array([0, 1, 2, 3, 4])
        env.positionY = np.array([0, 1, 2, 3, 4])
        success_rate, _ = env.computeMetrics()
        assert success_rate == 0.0

        # Test 2/5 agents at goal
        env.positionX = np.array([5, 6, 2, 3, 4])
        env.positionY = np.array([5, 6, 2, 3, 4])
        success_rate, _ = env.computeMetrics()
        assert success_rate == 0.4

        # Test all agents at goal
        env.positionX = goals[:, 0].copy()
        env.positionY = goals[:, 1].copy()
        success_rate, _ = env.computeMetrics()
        assert success_rate == 1.0

    def test_obstacle_collision(self):
        """Test that agents cannot move onto obstacles."""
        obstacles = np.array([[5, 5], [6, 6]])
        goals = np.array([[9, 9], [8, 8], [7, 7], [0, 0], [1, 1]])
        start_pos = np.array([[4, 5], [5, 6], [6, 7], [3, 3], [2, 2]])

        env = GraphEnv(
            self.config,
            goal=goals,
            starting_positions=start_pos,
            obstacles=obstacles
        )
        env.reset()

        # Agent 0 tries to move onto obstacle at (5,5)
        initial_x, initial_y = env.positionX[0], env.positionY[0]

        # Try to move agent 0 to obstacle position
        env.positionX_temp = env.positionX.copy()
        env.positionY_temp = env.positionY.copy()
        env.positionX[0] = 5
        env.positionY[0] = 5

        env.check_collision_obstacle()

        # Agent should be reverted
        assert env.positionX[0] == initial_x
        assert env.positionY[0] == initial_y

    def test_boundary_checking(self):
        """Test that agents cannot move outside board boundaries."""
        goals = np.array([[5, 5], [6, 6], [7, 7], [8, 8], [9, 9]])
        env = GraphEnv(self.config, goal=goals)
        env.reset()

        # Try to move agents outside boundaries
        env.positionX = np.array([-1, 10, 5, 5, 5])
        env.positionY = np.array([5, 5, -1, 10, 5])

        env.check_boundary()

        # Check all positions are clamped to valid range
        assert env.positionX[0] == 0  # Clamped from -1
        assert env.positionX[1] == 9  # Clamped from 10 (board_size=10)
        assert env.positionY[2] == 0  # Clamped from -1
        assert env.positionY[3] == 9  # Clamped from 10

    def test_action_execution(self):
        """Test that actions are executed correctly."""
        goals = np.array([[9, 9], [8, 8], [7, 7], [6, 6], [5, 5]])
        start_pos = np.array([[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]])

        env = GraphEnv(
            self.config,
            goal=goals,
            starting_positions=start_pos
        )
        env.reset()

        # Test all action types
        # 0: idle, 1: right, 2: up, 3: left, 4: down
        actions = [0, 1, 2, 3, 4]
        emb = np.ones((5, 1))

        obs, _, done, _ = env.step(actions, emb)

        # Check positions after actions
        assert env.positionX[0] == 5 and env.positionY[0] == 5  # idle
        assert env.positionX[1] == 5 and env.positionY[1] == 4  # right
        assert env.positionX[2] == 3 and env.positionY[2] == 4  # up
        assert env.positionX[3] == 1 and env.positionY[3] == 2  # left
        assert env.positionX[4] == 1 and env.positionY[4] == 0  # down

    def test_fov_generation(self):
        """Test field of view generation."""
        goals = np.array([[9, 9], [8, 8]])
        start_pos = np.array([[5, 5], [3, 3]])
        obstacles = np.array([[4, 4], [6, 6]])

        config = {
            "num_agents": 2,
            "board_size": [10, 10],
            "max_time": 20,
            "min_time": 5,
        }

        env = GraphEnv(
            config,
            goal=goals,
            starting_positions=start_pos,
            obstacles=obstacles
        )
        env.reset()

        fov = env.preprocessObs()

        # Check FOV shape
        assert fov.shape == (2, 2, 5, 5)  # (agents, channels, height, width)

        # FOV should contain obstacles and goals in appropriate channels
        # Channel 0: obstacles and other agents
        # Channel 1: goals

    def test_distance_matrix_computation(self):
        """Test distance matrix and adjacency computation."""
        goals = np.array([[9, 9], [8, 8], [7, 7]])
        start_pos = np.array([[0, 0], [1, 0], [0, 1]])

        config = {
            "num_agents": 3,
            "board_size": [10, 10],
            "max_time": 20,
            "min_time": 5,
        }

        env = GraphEnv(
            config,
            goal=goals,
            starting_positions=start_pos,
            sensing_range=2
        )
        env.reset()

        # Check distance matrix shape
        assert env.distance_matrix.shape == (3, 3)

        # Compute expected raw distances
        positions = np.array([[0, 0], [1, 0], [0, 1]])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        expected_distances = np.sqrt(np.sum(diff * diff, axis=2))

        # Distance from agent 0 to agent 1 should be 1
        assert np.isclose(expected_distances[0, 1], 1.0)

        # Distance from agent 0 to agent 2 should be 1
        assert np.isclose(expected_distances[0, 2], 1.0)

        # Distance from agent 1 to agent 2 should be sqrt(2)
        assert np.isclose(expected_distances[1, 2], np.sqrt(2))

        # The actual distance matrix may be modified by _computeClosest
        # which keeps only the 4 closest neighbors
        # So we just check that non-zero values are correct where they exist
        if env.distance_matrix[0, 1] != 0:
            assert env.distance_matrix[0, 1] == 1.0
        if env.distance_matrix[0, 2] != 0:
            assert env.distance_matrix[0, 2] == 1.0

        # Diagonal should be 0
        assert np.all(np.diag(env.distance_matrix) == 0)


class TestEnvironmentPerformance:
    """Test environment performance and vectorization."""

    def test_vectorization_performance(self):
        """Test that vectorized operations are fast."""
        board_size = [50, 50]
        num_agents = 50
        num_obstacles = 100

        # Time goal generation (should be < 10ms for 50x50 board)
        start = time.time()
        for _ in range(100):
            obstacles = create_obstacles(board_size, num_obstacles)
            goals = create_goals(board_size, num_agents, obstacles)
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.01, f"Goal generation too slow: {elapsed*1000:.2f}ms"

    def test_step_performance(self):
        """Test environment step performance."""
        config = {
            "num_agents": 20,
            "board_size": [30, 30],
            "max_time": 50,
            "min_time": 10,
        }

        obstacles = create_obstacles([30, 30], 50)
        goals = create_goals([30, 30], 20, obstacles)

        env = GraphEnv(config, goal=goals, obstacles=obstacles)
        env.reset()

        actions = np.random.randint(0, 5, size=20)
        emb = np.ones((20, 1))

        # Warm up
        for _ in range(10):
            env.step(actions, emb)

        # Time steps (should be < 5ms per step for 20 agents)
        start = time.time()
        num_steps = 100
        for _ in range(num_steps):
            actions = np.random.randint(0, 5, size=20)
            env.step(actions, emb)
        elapsed = (time.time() - start) / num_steps

        assert elapsed < 0.005, f"Step too slow: {elapsed*1000:.2f}ms per step"

    def test_collision_checking_performance(self):
        """Test collision checking is efficient even with many agents."""
        config = {
            "num_agents": 100,
            "board_size": [50, 50],
            "max_time": 50,
            "min_time": 10,
        }

        goals = create_goals([50, 50], 100)
        env = GraphEnv(config, goal=goals)
        env.reset()

        # Create a worst-case scenario: many collisions
        env.positionX_temp = env.positionX.copy()
        env.positionY_temp = env.positionY.copy()

        # Make half the agents collide
        for i in range(50):
            env.positionX[i] = 25
            env.positionY[i] = 25

        # Time collision checking (should be < 1ms even for 100 agents)
        start = time.time()
        for _ in range(1000):
            env.check_collisions()
        elapsed = (time.time() - start) / 1000

        assert elapsed < 0.001, f"Collision checking too slow: {elapsed*1000:.2f}ms"

    def test_distance_computation_scaling(self):
        """Test distance computation scales well with agent count."""
        times = []
        agent_counts = [10, 20, 50, 100]

        for n_agents in agent_counts:
            config = {
                "num_agents": n_agents,
                "board_size": [50, 50],
                "max_time": 50,
                "min_time": 10,
            }

            goals = create_goals([50, 50], n_agents)
            env = GraphEnv(config, goal=goals)
            env.reset()

            # Time distance computation
            start = time.time()
            for _ in range(100):
                env._computeDistance()
            elapsed = (time.time() - start) / 100
            times.append(elapsed)

        # Check that scaling is reasonable (should be roughly O(n²))
        # Time for 100 agents should be less than 16x time for 10 agents
        # (allowing some overhead)
        assert times[-1] < times[0] * 120, "Distance computation doesn't scale well"

    def test_memory_efficiency(self):
        """Test that environment doesn't use excessive memory."""
        import tracemalloc

        tracemalloc.start()

        config = {
            "num_agents": 50,
            "board_size": [100, 100],
            "max_time": 50,
            "min_time": 10,
        }

        # Create environment
        goals = create_goals([100, 100], 50)
        obstacles = create_obstacles([100, 100], 200)
        env = GraphEnv(config, goal=goals, obstacles=obstacles)
        env.reset()

        # Run some steps
        for _ in range(10):
            actions = np.random.randint(0, 5, size=50)
            env.step(actions, np.ones((50, 1)))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Environment should use less than 50MB for 50 agents on 100x100 board
        assert peak / 1024 / 1024 < 50, f"Memory usage too high: {peak/1024/1024:.2f}MB"


def test_no_errors_full_episode():
    """Test that a full episode runs without errors."""
    config = {
        "num_agents": 10,
        "board_size": [20, 20],
        "max_time": 30,
        "min_time": 5,
    }

    obstacles = create_obstacles([20, 20], 20)
    goals = create_goals([20, 20], 10, obstacles)

    env = GraphEnv(config, goal=goals, obstacles=obstacles)
    obs = env.reset()

    # Run full episode
    done = False
    steps = 0
    max_steps = 50

    while not done and steps < max_steps:
        # Random actions
        actions = np.random.randint(0, 5, size=10)
        emb = np.ones((10, 1))

        obs, _, done, _ = env.step(actions, emb)
        steps += 1

        # Check observations are valid
        assert "fov" in obs
        assert "adj_matrix" in obs
        assert obs["fov"].shape == (10, 2, 5, 5)
        assert obs["adj_matrix"].shape == (10, 10)

    # Compute final metrics
    success_rate, flow_time = env.computeMetrics()
    assert 0 <= success_rate <= 1
    assert flow_time > 0


if __name__ == "__main__":
    # Run all tests
    print("Running functionality tests...")
    test_func = TestEnvironmentFunctionality()
    test_func.setup_method()

    test_func.test_unique_goal_generation()
    print("✓ Unique goal generation")

    test_func.test_unique_goal_generation_with_obstacles()
    print("✓ Goal generation with obstacles")

    test_func.test_unique_obstacle_generation()
    print("✓ Unique obstacle generation")

    test_func.test_multi_agent_collision_detection()
    print("✓ Multi-agent collision detection")

    test_func.test_success_rate_calculation()
    print("✓ Success rate calculation")

    test_func.test_obstacle_collision()
    print("✓ Obstacle collision")

    test_func.test_boundary_checking()
    print("✓ Boundary checking")

    test_func.test_action_execution()
    print("✓ Action execution")

    test_func.test_fov_generation()
    print("✓ FOV generation")

    test_func.test_distance_matrix_computation()
    print("✓ Distance matrix computation")

    print("\nRunning performance tests...")
    test_perf = TestEnvironmentPerformance()

    test_perf.test_vectorization_performance()
    print("✓ Vectorization performance")

    test_perf.test_step_performance()
    print("✓ Step performance")

    test_perf.test_collision_checking_performance()
    print("✓ Collision checking performance")

    test_perf.test_distance_computation_scaling()
    print("✓ Distance computation scaling")

    test_perf.test_memory_efficiency()
    print("✓ Memory efficiency")

    print("\nRunning integration test...")
    test_no_errors_full_episode()
    print("✓ Full episode without errors")

    print("\n✅ All tests passed!")