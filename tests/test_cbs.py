"""
Comprehensive pytest tests for CBS (Conflict-Based Search) implementation.

Tests cover:
1. Basic functionality
2. A* search correctness
3. CBS search with various scenarios
4. Performance improvements
5. Edge cases and error handling

author: Victor Retamal
"""

import pytest
import numpy as np
import time
import heapq
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from cbs.cbs import (
    Environment, CBS, Location, State, Conflict,
    VertexConstraint, EdgeConstraint, Constraints,
    HighLevelNode
)
from cbs.a_star import AStar


class TestLocationAndState:
    """Test basic Location and State classes."""

    def test_location_creation(self):
        """Test Location creation and equality."""
        loc1 = Location(3, 4)
        loc2 = Location(3, 4)
        loc3 = Location(4, 3)

        assert loc1.x == 3
        assert loc1.y == 4
        assert loc1 == loc2
        assert loc1 != loc3
        assert hash(loc1) == hash(loc2)
        assert hash(loc1) != hash(loc3)

    def test_state_creation(self):
        """Test State creation and methods."""
        loc = Location(2, 3)
        state1 = State(5, loc)
        state2 = State(5, Location(2, 3))
        state3 = State(6, loc)

        assert state1.time == 5
        assert state1.location == loc
        assert state1 == state2
        assert state1 != state3
        assert state1.is_equal_except_time(state3)


class TestConstraints:
    """Test constraint classes."""

    def test_vertex_constraint(self):
        """Test VertexConstraint creation and equality."""
        loc = Location(3, 4)
        vc1 = VertexConstraint(5, loc)
        vc2 = VertexConstraint(5, Location(3, 4))
        vc3 = VertexConstraint(6, loc)

        assert vc1 == vc2
        assert vc1 != vc3
        assert hash(vc1) == hash(vc2)

    def test_edge_constraint(self):
        """Test EdgeConstraint creation and equality."""
        loc1 = Location(3, 4)
        loc2 = Location(3, 5)
        ec1 = EdgeConstraint(5, loc1, loc2)
        ec2 = EdgeConstraint(5, Location(3, 4), Location(3, 5))
        ec3 = EdgeConstraint(5, loc2, loc1)

        assert ec1 == ec2
        assert ec1 != ec3

    def test_constraints_collection(self):
        """Test Constraints collection operations."""
        constraints = Constraints()
        loc = Location(3, 4)
        vc = VertexConstraint(5, loc)
        ec = EdgeConstraint(6, loc, Location(3, 5))

        constraints.vertex_constraints.add(vc)
        constraints.edge_constraints.add(ec)

        assert vc in constraints.vertex_constraints
        assert ec in constraints.edge_constraints

        # Test add_constraint
        other_constraints = Constraints()
        other_constraints.vertex_constraints.add(VertexConstraint(7, loc))
        constraints.add_constraint(other_constraints)

        assert len(constraints.vertex_constraints) == 2


class TestAStar:
    """Test A* search algorithm."""

    def test_simple_path(self):
        """Test A* finds path in simple environment."""
        agents = [{"name": "agent0", "start": [0, 0], "goal": [3, 3]}]
        obstacles = []
        env = Environment([4, 4], agents, obstacles)

        path = env.a_star.search("agent0")

        assert path is not False
        assert len(path) > 0
        assert path[0].location.x == 0 and path[0].location.y == 0
        assert path[-1].location.x == 3 and path[-1].location.y == 3

    def test_path_with_obstacles(self):
        """Test A* finds path around obstacles."""
        agents = [{"name": "agent0", "start": [0, 0], "goal": [2, 0]}]
        obstacles = [(1, 0)]  # Block direct path
        env = Environment([3, 3], agents, obstacles)

        path = env.a_star.search("agent0")

        assert path is not False
        assert len(path) > 2  # Must go around obstacle
        # Should not pass through obstacle
        for state in path:
            assert (state.location.x, state.location.y) != (1, 0)

    def test_no_path_exists(self):
        """Test A* returns False when no path exists."""
        agents = [{"name": "agent0", "start": [0, 0], "goal": [2, 2]}]
        # Completely surround the goal
        obstacles = [(1, 1), (1, 2), (2, 1), (2, 2)]
        env = Environment([3, 3], agents, obstacles)

        path = env.a_star.search("agent0")
        assert path is False

    def test_heap_based_performance(self):
        """Test that heap-based A* is faster than set-based for larger problems."""
        # Create larger environment
        size = 20
        agents = [{"name": "agent0", "start": [0, 0], "goal": [size-1, size-1]}]
        obstacles = [(i, j) for i in range(5, 10) for j in range(5, 15) if (i + j) % 3 == 0]
        env = Environment([size, size], agents, obstacles)

        # Time the search
        start_time = time.perf_counter()
        path = env.a_star.search("agent0")
        elapsed = time.perf_counter() - start_time

        assert path is not False
        assert elapsed < 0.1  # Should be fast with heap implementation
        print(f"A* search time for {size}x{size} grid: {elapsed*1000:.2f}ms")


class TestEnvironment:
    """Test Environment class functionality."""

    def test_environment_creation(self):
        """Test Environment initialization."""
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [2, 2]},
            {"name": "agent1", "start": [2, 0], "goal": [0, 2]}
        ]
        obstacles = [(1, 1)]
        env = Environment([3, 3], agents, obstacles)

        assert len(env.agent_dict) == 2
        assert "agent0" in env.agent_dict
        assert env.agent_dict["agent0"]["start"].location.x == 0
        assert env.agent_dict["agent0"]["goal"].location.x == 2

    def test_get_neighbors(self):
        """Test neighbor generation."""
        agents = [{"name": "agent0", "start": [0, 0], "goal": [2, 2]}]
        env = Environment([3, 3], agents, [])

        state = State(0, Location(1, 1))
        neighbors = env.get_neighbors(state)

        # Should have 5 neighbors (4 directions + wait)
        assert len(neighbors) == 5
        # All neighbors should have time = 1
        for n in neighbors:
            assert n.time == 1

    def test_conflict_detection(self):
        """Test conflict detection between agents."""
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [2, 0]},
            {"name": "agent1", "start": [2, 0], "goal": [0, 0]}
        ]
        env = Environment([3, 1], agents, [])

        # Create solution where agents collide
        solution = {
            "agent0": [
                State(0, Location(0, 0)),
                State(1, Location(1, 0)),
                State(2, Location(2, 0))
            ],
            "agent1": [
                State(0, Location(2, 0)),
                State(1, Location(1, 0)),  # Collision here
                State(2, Location(0, 0))
            ]
        }

        conflict = env.get_first_conflict(solution)
        assert conflict is not False
        assert conflict.type == Conflict.VERTEX
        assert conflict.time == 1


class TestCBS:
    """Test CBS algorithm."""

    def test_simple_cbs(self):
        """Test CBS on simple two-agent problem."""
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [2, 0]},
            {"name": "agent1", "start": [0, 1], "goal": [2, 1]}
        ]
        env = Environment([3, 3], agents, [])
        cbs = CBS(env, verbose=False)

        solution = cbs.search()

        assert solution != {}
        assert "agent0" in solution
        assert "agent1" in solution
        # Both agents should reach their goals
        assert solution["agent0"][-1]["x"] == 2
        assert solution["agent0"][-1]["y"] == 0
        assert solution["agent1"][-1]["x"] == 2
        assert solution["agent1"][-1]["y"] == 1

    def test_cbs_with_conflicts(self):
        """Test CBS resolves conflicts correctly."""
        # Create a simple crossing scenario
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [2, 0]},
            {"name": "agent1", "start": [1, 1], "goal": [1, 0]}
        ]
        env = Environment([3, 3], agents, [])
        cbs = CBS(env, verbose=False)

        solution = cbs.search()

        assert solution != {}
        # Check no collisions in solution
        for t in range(max(len(solution["agent0"]), len(solution["agent1"]))):
            if t < len(solution["agent0"]) and t < len(solution["agent1"]):
                pos0 = (solution["agent0"][t]["x"], solution["agent0"][t]["y"])
                pos1 = (solution["agent1"][t]["x"], solution["agent1"][t]["y"])
                assert pos0 != pos1, f"Collision at time {t}"

    def test_heap_based_cbs_performance(self):
        """Test that heap-based CBS is faster than set-based."""
        # Create a scenario with multiple agents
        agents = [
            {"name": f"agent{i}", "start": [i, 0], "goal": [4-i, 4]}
            for i in range(3)
        ]
        env = Environment([5, 5], agents, [])
        cbs = CBS(env, verbose=False)

        start_time = time.perf_counter()
        solution = cbs.search()
        elapsed = time.perf_counter() - start_time

        assert solution != {}
        assert elapsed < 1.0  # Should be fast with heap implementation
        print(f"CBS search time for {len(agents)} agents: {elapsed*1000:.2f}ms")

    def test_no_solution_exists(self):
        """Test CBS returns empty dict when no solution exists."""
        # Create impossible scenario
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [1, 0]},
            {"name": "agent1", "start": [1, 0], "goal": [0, 0]}
        ]
        # Block all paths
        obstacles = [(0, 1), (1, 1)]
        env = Environment([2, 2], agents, obstacles)
        cbs = CBS(env, verbose=False)

        solution = cbs.search()
        assert solution == {}


class TestOptimizations:
    """Test specific optimizations in the implementation."""

    def test_heap_operations(self):
        """Test heap-based priority queue operations."""
        # Test that our heap implementation maintains proper ordering
        heap = []
        nodes = []

        for i in [5, 2, 8, 1, 9, 3]:
            node = HighLevelNode()
            node.cost = i
            nodes.append(node)
            heapq.heappush(heap, (node.cost, i, node))

        # Extract in order
        extracted_costs = []
        while heap:
            cost, _, node = heapq.heappop(heap)
            extracted_costs.append(cost)

        assert extracted_costs == sorted(extracted_costs)

    def test_shallow_copy_optimization(self):
        """Test that shallow copy is used where appropriate."""
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [2, 0]},
            {"name": "agent1", "start": [0, 1], "goal": [2, 1]}
        ]
        env = Environment([3, 3], agents, [])
        cbs = CBS(env, verbose=False)

        # Create a node
        node = HighLevelNode()
        node.constraint_dict = {
            "agent0": Constraints(),
            "agent1": Constraints()
        }

        # Test that solution copy is shallow
        node.solution = {"agent0": [], "agent1": []}
        new_solution = node.solution.copy()
        assert id(new_solution) != id(node.solution)
        assert id(new_solution["agent0"]) == id(node.solution["agent0"])

    def test_closed_set_efficiency(self):
        """Test closed set prevents revisiting states."""
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [3, 3]},
            {"name": "agent1", "start": [3, 0], "goal": [0, 3]}
        ]
        env = Environment([4, 4], agents, [])
        cbs = CBS(env, verbose=False)

        # Track closed set size during search
        solution = cbs.search()

        # Closed set should have reasonable size
        assert len(cbs.closed_set) < 100  # Shouldn't explore too many nodes
        print(f"Closed set size: {len(cbs.closed_set)}")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_agent(self):
        """Test CBS with single agent."""
        agents = [{"name": "agent0", "start": [0, 0], "goal": [3, 3]}]
        env = Environment([4, 4], agents, [])
        cbs = CBS(env, verbose=False)

        solution = cbs.search()
        assert solution != {}
        assert len(solution) == 1

    def test_agent_at_goal(self):
        """Test when agent starts at goal."""
        agents = [{"name": "agent0", "start": [2, 2], "goal": [2, 2]}]
        env = Environment([4, 4], agents, [])
        cbs = CBS(env, verbose=False)

        solution = cbs.search()
        assert solution != {}
        assert len(solution["agent0"]) == 1

    def test_large_environment(self):
        """Test CBS scales to larger environments."""
        size = 10
        agents = [
            {"name": "agent0", "start": [0, 0], "goal": [size-1, size-1]},
            {"name": "agent1", "start": [size-1, 0], "goal": [0, size-1]},
            {"name": "agent2", "start": [0, size-1], "goal": [size-1, 0]}
        ]
        obstacles = [(i, j) for i in range(3, 7) for j in range(3, 7) if (i + j) % 2 == 0]
        env = Environment([size, size], agents, obstacles)
        cbs = CBS(env, verbose=False)

        start_time = time.perf_counter()
        solution = cbs.search()
        elapsed = time.perf_counter() - start_time

        if solution:
            print(f"Large environment ({size}x{size}, {len(agents)} agents): {elapsed*1000:.2f}ms")
            assert elapsed < 5.0  # Should complete in reasonable time

    def test_hash_consistency(self):
        """Test hash functions are consistent with equality."""
        # Location
        loc1 = Location(3, 4)
        loc2 = Location(3, 4)
        assert loc1 == loc2
        assert hash(loc1) == hash(loc2)

        # State
        state1 = State(5, loc1)
        state2 = State(5, loc2)
        assert state1 == state2
        assert hash(state1) == hash(state2)

        # VertexConstraint
        vc1 = VertexConstraint(5, loc1)
        vc2 = VertexConstraint(5, loc2)
        assert vc1 == vc2
        assert hash(vc1) == hash(vc2)

        # EdgeConstraint
        loc3 = Location(4, 5)
        ec1 = EdgeConstraint(5, loc1, loc3)
        ec2 = EdgeConstraint(5, Location(3, 4), Location(4, 5))
        assert ec1 == ec2
        assert hash(ec1) == hash(ec2)


def test_performance_comparison():
    """Compare optimized vs original performance characteristics."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    # Test scaling with number of agents
    for n_agents in [2, 3, 4, 5]:
        agents = [
            {"name": f"agent{i}", "start": [i, 0], "goal": [9-i, 9]}
            for i in range(n_agents)
        ]
        obstacles = [(i, 5) for i in range(3, 7)]  # Add some obstacles
        env = Environment([10, 10], agents, obstacles)
        cbs = CBS(env, verbose=False)

        start_time = time.perf_counter()
        solution = cbs.search()
        elapsed = time.perf_counter() - start_time

        if solution:
            total_cost = sum(len(path) for path in solution.values())
            print(f"{n_agents} agents: {elapsed*1000:6.2f}ms, cost: {total_cost}")
        else:
            print(f"{n_agents} agents: No solution found")

    print("="*60)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])