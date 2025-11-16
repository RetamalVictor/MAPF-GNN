"""
Test CBS performance and verify identified issues.
"""

import time
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from cbs.cbs import Environment, CBS
from cbs.a_star import AStar


def test_cbs_performance():
    """Test CBS performance with different problem sizes."""
    print("\n" + "="*60)
    print("CBS PERFORMANCE ANALYSIS")
    print("="*60)

    # Test cases: (dimension, num_agents, num_obstacles)
    test_cases = [
        (10, 2, 5),   # Small
        (10, 3, 5),   # Small with more agents
        (15, 4, 10),  # Medium
        (20, 5, 15),  # Larger
    ]

    results = []

    for dim, n_agents, n_obstacles in test_cases:
        print(f"\nTest: {dim}x{dim} grid, {n_agents} agents, {n_obstacles} obstacles")

        # Generate random problem
        obstacles = []
        for _ in range(n_obstacles):
            while True:
                obs = (np.random.randint(0, dim), np.random.randint(0, dim))
                if obs not in obstacles:
                    obstacles.append(obs)
                    break

        # Generate agent starts and goals
        agents = []
        positions_used = set(obstacles)

        for i in range(n_agents):
            # Find valid start
            while True:
                start = (np.random.randint(0, dim), np.random.randint(0, dim))
                if start not in positions_used:
                    positions_used.add(start)
                    break

            # Find valid goal
            while True:
                goal = (np.random.randint(0, dim), np.random.randint(0, dim))
                if goal not in positions_used:
                    positions_used.add(goal)
                    break

            agents.append({
                "name": f"agent{i}",
                "start": list(start),
                "goal": list(goal)
            })

        # Create environment and CBS
        env = Environment([dim, dim], agents, obstacles)
        cbs = CBS(env)

        # Time the search
        start_time = time.perf_counter()
        solution = cbs.search()
        elapsed = time.perf_counter() - start_time

        # Check if solution found
        if solution:
            cost = sum(len(path) for path in solution.values())
            print(f"  ✓ Solution found in {elapsed:.3f}s")
            print(f"    Total path cost: {cost}")
            print(f"    Nodes explored: {len(cbs.closed_set)}")
        else:
            print(f"  ✗ No solution found after {elapsed:.3f}s")

        results.append({
            "dimension": dim,
            "agents": n_agents,
            "obstacles": n_obstacles,
            "time": elapsed,
            "nodes": len(cbs.closed_set) if solution else -1,
            "solution": bool(solution)
        })

    return results


def test_open_set_performance():
    """Demonstrate the O(n) open set performance issue."""
    print("\n" + "="*60)
    print("OPEN SET PERFORMANCE ISSUE DEMONSTRATION")
    print("="*60)

    # Create a simple mock of the CBS open set behavior
    class MockNode:
        def __init__(self, cost):
            self.cost = cost

        def __lt__(self, other):
            return self.cost < other.cost

        def __hash__(self):
            return hash(self.cost)

        def __eq__(self, other):
            return self.cost == other.cost

    print("\nComparing set-based min() vs heap-based extraction:")

    sizes = [10, 50, 100, 500, 1000]

    for size in sizes:
        # Set-based approach (current CBS implementation)
        open_set = set()
        for i in range(size):
            open_set.add(MockNode(np.random.randint(0, 1000)))

        start = time.perf_counter()
        for _ in range(100):
            if open_set:
                min_node = min(open_set)
                open_set.remove(min_node)
                open_set.add(MockNode(np.random.randint(0, 1000)))
        set_time = (time.perf_counter() - start) / 100

        # Heap-based approach (optimized)
        import heapq
        open_heap = []
        for i in range(size):
            heapq.heappush(open_heap, (np.random.randint(0, 1000), i))

        start = time.perf_counter()
        counter = size
        for _ in range(100):
            if open_heap:
                _, _ = heapq.heappop(open_heap)
                heapq.heappush(open_heap, (np.random.randint(0, 1000), counter))
                counter += 1
        heap_time = (time.perf_counter() - start) / 100

        speedup = set_time / heap_time
        print(f"  Size {size:4d}: Set {set_time*1000:.3f}ms, Heap {heap_time*1000:.3f}ms, Speedup: {speedup:.1f}x")


def test_deepcopy_performance():
    """Demonstrate the deep copy performance issue."""
    print("\n" + "="*60)
    print("DEEP COPY PERFORMANCE ISSUE DEMONSTRATION")
    print("="*60)

    from copy import deepcopy

    # Create a mock node with constraints
    class MockConstraints:
        def __init__(self, size):
            self.vertex_constraints = set(range(size))
            self.edge_constraints = set(range(size))

    class MockNode:
        def __init__(self, num_agents, num_constraints):
            self.constraint_dict = {}
            for i in range(num_agents):
                self.constraint_dict[f"agent{i}"] = MockConstraints(num_constraints)
            self.cost = 0
            self.solution = {}

    print("\nDeep copy time for different node sizes:")

    test_cases = [
        (5, 10),   # 5 agents, 10 constraints each
        (10, 20),  # 10 agents, 20 constraints
        (20, 50),  # 20 agents, 50 constraints
        (50, 100), # 50 agents, 100 constraints
    ]

    for num_agents, num_constraints in test_cases:
        node = MockNode(num_agents, num_constraints)

        start = time.perf_counter()
        for _ in range(100):
            new_node = deepcopy(node)
        elapsed = (time.perf_counter() - start) / 100

        print(f"  {num_agents:2d} agents, {num_constraints:3d} constraints: {elapsed*1000:.3f}ms per copy")


def test_conflict_detection_scaling():
    """Test conflict detection performance scaling."""
    print("\n" + "="*60)
    print("CONFLICT DETECTION SCALING")
    print("="*60)

    from itertools import combinations

    print("\nTime to check all agent pairs for conflicts:")

    for n_agents in [5, 10, 20, 50, 100]:
        agents = [f"agent{i}" for i in range(n_agents)]

        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            for agent_1, agent_2 in combinations(agents, 2):
                # Simulate conflict check
                _ = agent_1 < agent_2
        elapsed = (time.perf_counter() - start) / iterations

        num_pairs = n_agents * (n_agents - 1) // 2
        print(f"  {n_agents:3d} agents ({num_pairs:4d} pairs): {elapsed*1000:.3f}ms")


def analyze_data_generation_issues():
    """Demonstrate data generation pipeline issues."""
    print("\n" + "="*60)
    print("DATA GENERATION ISSUES")
    print("="*60)

    print("\n1. Path separator issues:")
    bad_path = r"dataset\5_8_28\train"
    print(f"   Windows path: {bad_path}")
    print(f"   Will fail on Unix/Mac")

    good_path = os.path.join("dataset", "5_8_28", "train")
    print(f"   Cross-platform: {good_path}")

    print("\n2. Random assignment efficiency (current approach):")

    for fill_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
        board_size = 20
        total_cells = board_size * board_size
        filled_cells = int(total_cells * fill_ratio)

        obstacles = set()
        attempts = 0
        start = time.perf_counter()

        while len(obstacles) < filled_cells:
            attempts += 1
            pos = (np.random.randint(0, board_size), np.random.randint(0, board_size))
            obstacles.add(pos)

            if attempts > filled_cells * 10:  # Safety limit
                break

        elapsed = time.perf_counter() - start
        efficiency = filled_cells / attempts if attempts > 0 else 0

        print(f"   Fill {fill_ratio*100:.0f}%: {attempts} attempts for {filled_cells} positions")
        print(f"            Efficiency: {efficiency:.2%}, Time: {elapsed*1000:.2f}ms")


if __name__ == "__main__":
    print("CBS IMPLEMENTATION ANALYSIS")
    print("="*60)

    # Test CBS performance
    results = test_cbs_performance()

    # Demonstrate specific issues
    test_open_set_performance()
    test_deepcopy_performance()
    test_conflict_detection_scaling()
    analyze_data_generation_issues()

    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    print("1. ✗ O(n) open set operations cause quadratic complexity")
    print("2. ✗ Deep copying entire nodes is memory inefficient")
    print("3. ✗ Conflict detection scales poorly with agent count")
    print("4. ✗ Data generation has cross-platform path issues")
    print("5. ✗ Random assignment becomes inefficient as board fills")
    print("\nThese issues significantly impact scalability for larger MAPF instances.")