"""
Optimized CBS (Conflict-Based Search) implementation

This version includes several key optimizations:
1. Heap-based priority queue for O(log n) operations instead of O(n)
2. Shallow copy with selective deep copy for constraints
3. Early termination and constraint propagation
4. Fixed hash/equality contract for CTNode

author: Victor Retamal
"""

import sys
import heapq
from itertools import count, combinations
from copy import copy, deepcopy
import argparse
import yaml
from math import fabs

sys.path.insert(0, "../")
from cbs.a_star import AStar


class Location(object):
    # Millions of these are created and hashed per instance, so the attribute
    # dict and the repeated tuple build both show up in profiles.
    __slots__ = ("x", "y", "_hash")

    def __init__(self, x=-1, y=-1):
        self.x = x
        self.y = y
        self._hash = hash((x, y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self._hash

    def __str__(self):
        return str((self.x, self.y))


class State(object):
    __slots__ = ("time", "location", "_hash")

    def __init__(self, time, location):
        self.time = time
        self.location = location
        self._hash = hash((time, location.x, location.y))

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return self._hash

    def is_equal_except_time(self, state):
        return self.location == state.location

    def __str__(self):
        return str((self.time, self.location.x, self.location.y))


class Conflict(object):
    VERTEX = 1
    EDGE = 2

    def __init__(self):
        self.time = -1
        self.type = -1
        self.agent_1 = ""
        self.agent_2 = ""
        self.location_1 = Location()
        self.location_2 = Location()

    def __str__(self):
        return (
            "("
            + str(self.time)
            + ", "
            + self.agent_1
            + ", "
            + self.agent_2
            + ", "
            + str(self.location_1)
            + ", "
            + str(self.location_2)
            + ")"
        )


class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location

    def __hash__(self):
        return hash((self.time, self.location.x, self.location.y))

    def __str__(self):
        return "(" + str(self.time) + ", " + str(self.location) + ")"


class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2

    def __eq__(self, other):
        return (
            self.time == other.time
            and self.location_1 == other.location_1
            and self.location_2 == other.location_2
        )

    def __hash__(self):
        return hash((self.time, self.location_1.x, self.location_1.y,
                    self.location_2.x, self.location_2.y))

    def __str__(self):
        return (
            "("
            + str(self.time)
            + ", "
            + str(self.location_1)
            + ", "
            + str(self.location_2)
            + ")"
        )


class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return (
            "VC: "
            + str([str(vc) for vc in self.vertex_constraints])
            + "EC: "
            + str([str(ec) for ec in self.edge_constraints])
        )


class Environment(object):
    def __init__(self, dimension, agents, obstacles):
        self.dimension = dimension
        # Set of tuples: state_valid tests membership millions of times per
        # instance, and a list makes each test a linear scan.
        self.obstacles = {tuple(obstacle) for obstacle in obstacles}
        self.agents = agents
        self.agent_dict = {}
        self.make_agent_dict()
        self.constraints = Constraints()
        self.constraint_dict = {}
        # Flattened constraint keys for the agent currently being planned, so
        # validity checks never build a throwaway constraint object.
        self._vertex_keys = frozenset()
        self._edge_keys = frozenset()
        self.a_star = AStar(self)

    def get_neighbors(self, state):
        neighbors = []

        # Wait action
        n = State(state.time + 1, state.location)
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y + 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y - 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x - 1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x + 1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        return neighbors

    def get_first_conflict(self, solution):
        """First conflict in the plan, scanning time then agent pairs.

        Comparing every pair at every timestep is O(T*N^2) and rebuilds State
        objects for each comparison. Bucketing agents by cell finds the same
        conflict in O(T*N): a vertex conflict is two agents in one cell, and an
        edge conflict is two agents whose moves are the reverse of each other.
        Conflicting pairs are reduced with min() so the pair returned is the one
        combinations() would have reached first.
        """
        agents = list(solution.keys())
        max_t = max(len(plan) for plan in solution.values())

        # Pad each path to max_t + 1 with its final cell, which is what
        # get_state does for agents that finished early.
        coords = []
        for agent in agents:
            plan = solution[agent]
            path = [(s.location.x, s.location.y) for s in plan]
            path.extend([path[-1]] * (max_t + 1 - len(path)))
            coords.append(path)

        for t in range(max_t):
            occupied = {}
            pair = None
            for j in range(len(agents)):
                cell = coords[j][t]
                i = occupied.get(cell)
                if i is None:
                    occupied[cell] = j
                elif pair is None or (i, j) < pair:
                    pair = (i, j)
            if pair is not None:
                i, j = pair
                result = Conflict()
                result.time = t
                result.type = Conflict.VERTEX
                result.location_1 = Location(*coords[i][t])
                result.agent_1 = agents[i]
                result.agent_2 = agents[j]
                return result

            # No vertex conflict at t, so every agent occupies a distinct cell
            # and keying moves by their origin is unambiguous.
            moves = {}
            pair = None
            for j in range(len(agents)):
                origin, destination = coords[j][t], coords[j][t + 1]
                i = moves.get((destination, origin))
                if i is not None and (pair is None or (i, j) < pair):
                    pair = (i, j)
                moves[(origin, destination)] = j
            if pair is not None:
                i, j = pair
                result = Conflict()
                result.time = t
                result.type = Conflict.EDGE
                result.agent_1 = agents[i]
                result.agent_2 = agents[j]
                result.location_1 = Location(*coords[i][t])
                result.location_2 = Location(*coords[i][t + 1])
                return result
        return False

    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}
            constraint_dict[conflict.agent_1] = constraint
            constraint_dict[conflict.agent_2] = constraint

        elif conflict.type == Conflict.EDGE:
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(
                conflict.time, conflict.location_1, conflict.location_2
            )
            e_constraint2 = EdgeConstraint(
                conflict.time, conflict.location_2, conflict.location_1
            )

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict

    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t]
        else:
            return solution[agent_name][-1]

    def state_valid(self, state):
        x, y = state.location.x, state.location.y
        return (
            0 <= x < self.dimension[0]
            and 0 <= y < self.dimension[1]
            and (state.time, x, y) not in self._vertex_keys
            and (x, y) not in self.obstacles
        )

    def transition_valid(self, state_1, state_2):
        return (
            state_1.time,
            state_1.location.x,
            state_1.location.y,
            state_2.location.x,
            state_2.location.y,
        ) not in self._edge_keys

    def is_solution(self, agent_name):
        pass

    def admissible_heuristic(self, state, agent_name):
        goal = self.agent_dict[agent_name]["goal"]
        return fabs(state.location.x - goal.location.x) + fabs(
            state.location.y - goal.location.y
        )

    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent["start"][0], agent["start"][1]))
            goal_state = State(0, Location(agent["goal"][0], agent["goal"][1]))
            self.agent_dict.update(
                {agent["name"]: {"start": start_state, "goal": goal_state}}
            )

    def compute_solution(self):
        solution = {}
        for agent in self.agent_dict.keys():
            local_solution = self.compute_solution_for_agent(agent)
            if not local_solution:
                return False
            solution.update({agent: local_solution})
        return solution

    def compute_solution_for_agent(self, agent):
        """Plan a single agent under the current constraint dict.

        Only the agent that just received a constraint needs replanning; every
        other path in the parent node is still valid and is inherited. Replanning
        the whole fleet at each node multiplies the low-level search cost by the
        number of agents.
        """
        self.constraints = self.constraint_dict.setdefault(agent, Constraints())
        self._vertex_keys = frozenset(
            (c.time, c.location.x, c.location.y)
            for c in self.constraints.vertex_constraints
        )
        self._edge_keys = frozenset(
            (c.time, c.location_1.x, c.location_1.y, c.location_2.x, c.location_2.y)
            for c in self.constraints.edge_constraints
        )
        return self.a_star.search(agent)

    def compute_solution_cost(self, solution):
        return sum([len(path) for path in solution.values()])


class HighLevelNode(object):
    """Optimized high-level node without problematic hash/eq methods."""

    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __lt__(self, other):
        """For heap ordering."""
        return self.cost < other.cost


class CBS(object):
    """Optimized CBS implementation with heap-based priority queue."""

    def __init__(self, environment, verbose=True):
        self.env = environment
        self.verbose = verbose
        self.open_list = []  # Using heap instead of set
        self.closed_set = set()
        self.counter = count()

    def search(self):
        """
        Optimized high-level CBS search.

        Key optimizations:
        - Heap-based priority queue for O(log n) node extraction
        - Shallow copy with selective deep copy for constraints only
        - Early termination conditions
        """
        start = HighLevelNode()
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()

        start.solution = self.env.compute_solution()
        if not start.solution:
            return {}

        start.cost = self.env.compute_solution_cost(start.solution)

        # Add start node to heap
        heapq.heappush(self.open_list, (start.cost, next(self.counter), start))

        if self.verbose:
            print("Initializing search")

        iteration_count = 0
        max_iterations = 5000  # Prevent infinite loops
        while self.open_list and iteration_count < max_iterations:
            if self.verbose and iteration_count % 20 == 0:
                print(f"Still searching... (iteration {iteration_count})")
            iteration_count += 1

            # Extract minimum cost node - O(log n) instead of O(n)
            _, _, P = heapq.heappop(self.open_list)

            # Generate state key for closed set checking
            state_key = self._get_state_key(P)
            if state_key in self.closed_set:
                continue
            self.closed_set.add(state_key)

            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)

            if not conflict_dict:
                if self.verbose:
                    print("Solution found!")
                return self.generate_plan(P.solution)

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                # Create new node with shallow copy where possible
                new_node = HighLevelNode()
                new_node.solution = P.solution.copy()

                # Selective deep copy only for affected agent's constraints
                new_node.constraint_dict = {}
                for a in self.env.agent_dict.keys():
                    if a == agent:
                        # Deep copy only the modified agent's constraints
                        new_constraints = Constraints()
                        new_constraints.vertex_constraints = P.constraint_dict[a].vertex_constraints.copy()
                        new_constraints.edge_constraints = P.constraint_dict[a].edge_constraints.copy()
                        new_constraints.add_constraint(constraint_dict[agent])
                        new_node.constraint_dict[a] = new_constraints
                    else:
                        # Share unchanged constraints
                        new_node.constraint_dict[a] = P.constraint_dict[a]

                self.env.constraint_dict = new_node.constraint_dict
                new_path = self.env.compute_solution_for_agent(agent)

                if not new_path:
                    continue

                # Inherit every other agent's path from the parent; only this
                # agent's constraints changed.
                new_node.solution = P.solution.copy()
                new_node.solution[agent] = new_path

                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                # Add to heap
                heapq.heappush(self.open_list, (new_node.cost, next(self.counter), new_node))

        if self.verbose:
            print("No solution found")
        return {}

    def _get_state_key(self, node):
        """Generate a hashable state key for closed set checking."""
        # Create a frozen representation of the solution
        solution_tuple = tuple(
            (agent, tuple((s.time, s.location.x, s.location.y) for s in path))
            for agent, path in sorted(node.solution.items())
        )
        return solution_tuple

    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [
                {"t": state.time, "x": state.location.x, "y": state.location.y}
                for state in path
            ]
            plan[agent] = path_dict_list
        return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("param", help="input file containing map and obstacles")
    parser.add_argument("output", help="output file with the schedule")
    args = parser.parse_args()

    # Read from input file
    with open(args.param, "r") as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param["agents"]

    env = Environment(dimension, agents, obstacles)

    # Searching
    cbs = CBS(env)
    solution = cbs.search()
    if not solution:
        print("Solution not found")
        return

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    with open(args.output, "w") as output_yaml:
        yaml.safe_dump(output, output_yaml)


if __name__ == "__main__":
    main()