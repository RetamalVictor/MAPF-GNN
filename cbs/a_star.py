"""
Optimized AStar search with heap-based priority queue

This version replaces O(n) set operations with O(log n) heap operations
for significant performance improvements on larger problems.

author: Victor Retamal
"""

import heapq
from itertools import count


class AStar:
    def __init__(self, env):
        self.agent_dict = env.agent_dict
        self.admissible_heuristic = env.admissible_heuristic
        self.is_at_goal = env.is_at_goal
        self.get_neighbors = env.get_neighbors

    def reconstruct_path(self, came_from, current):
        """Reconstruct path from came_from dictionary."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def search(self, agent_name):
        """
        Optimized low-level A* search using heap-based priority queue.

        Key optimizations:
        - Uses heapq for O(log n) min extraction instead of O(n) set operations
        - Maintains separate open_set for O(1) membership testing
        - Uses counter to ensure unique heap entries (handles ties)
        """
        initial_state = self.agent_dict[agent_name]["start"]
        step_cost = 1

        # Use a counter to break ties and ensure FIFO for equal f-scores
        counter = count()

        # Priority queue: (f_score, count, state)
        open_heap = []

        # Set for O(1) membership testing
        open_set = {initial_state}
        closed_set = set()

        came_from = {}
        g_score = {initial_state: 0}

        # Initialize with starting state
        f_initial = self.admissible_heuristic(initial_state, agent_name)
        heapq.heappush(open_heap, (f_initial, next(counter), initial_state))

        # Add iteration limit to prevent infinite loops
        max_iterations = 10000
        iterations = 0

        while open_heap and iterations < max_iterations:
            iterations += 1
            # Extract minimum f-score state - O(log n)
            current_f, _, current = heapq.heappop(open_heap)

            # Skip if we've already processed this state
            # (can happen with duplicate entries in heap)
            if current not in open_set:
                continue

            open_set.remove(current)

            # Check if we've reached the goal
            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path(came_from, current)

            closed_set.add(current)

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + step_cost

                # If we found a better path to neighbor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.admissible_heuristic(neighbor, agent_name)

                    # Add to open set if not already there
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                        heapq.heappush(open_heap, (f_score, next(counter), neighbor))
                    else:
                        # Already in open set with worse score, add new entry
                        # (old entry will be skipped when popped)
                        heapq.heappush(open_heap, (f_score, next(counter), neighbor))

        return False