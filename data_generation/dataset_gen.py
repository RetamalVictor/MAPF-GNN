"""
Dataset generation for MAPF training.
Generates random MAPF instances and solves them using CBS.

author: Victor Retamal
"""
import sys
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import yaml
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cbs.cbs import Environment, CBS


def gen_input(
    dimensions: Tuple[int, int],
    nb_obstacles: int,
    nb_agents: int,
    max_attempts: int = 10000
) -> Optional[Dict]:
    """
    Generate random MAPF instance with agents and obstacles.

    Args:
        dimensions: (width, height) of the grid
        nb_obstacles: Number of obstacles to place
        nb_agents: Number of agents
        max_attempts: Maximum attempts for placement before giving up

    Returns:
        Dictionary with agent and map configuration, or None if generation fails
    """
    input_dict = {
        "agents": [],
        "map": {
            "dimensions": list(dimensions),
            "obstacles": []
        }
    }

    total_cells = dimensions[0] * dimensions[1]
    required_cells = nb_obstacles + 2 * nb_agents  # obstacles + starts + goals

    if required_cells > total_cells * 0.9:
        print(f"Warning: Requesting {required_cells} positions in {total_cells} cells (>90% fill)")

    # Use set for O(1) lookup
    occupied_positions = set()

    def get_random_position(exclude_set: set, max_attempts: int = 1000) -> Optional[Tuple[int, int]]:
        """Get random position not in exclude set."""
        # For sparse boards, use random sampling
        if len(exclude_set) < total_cells * 0.7:
            for _ in range(max_attempts):
                pos = (
                    np.random.randint(0, dimensions[0]),
                    np.random.randint(0, dimensions[1])
                )
                if pos not in exclude_set:
                    return pos
        else:
            # For dense boards, sample from available positions
            all_positions = {(x, y) for x in range(dimensions[0]) for y in range(dimensions[1])}
            available = list(all_positions - exclude_set)
            if available:
                return available[np.random.randint(0, len(available))]

        return None

    # Place obstacles
    obstacles = []
    for _ in range(nb_obstacles):
        obs_pos = get_random_position(occupied_positions)
        if obs_pos is None:
            print(f"Failed to place all obstacles (placed {len(obstacles)}/{nb_obstacles})")
            return None

        obstacles.append(obs_pos)
        occupied_positions.add(obs_pos)
        input_dict["map"]["obstacles"].append(obs_pos)

    # Place agents
    for agent_id in range(nb_agents):
        # Get start position
        start_pos = get_random_position(occupied_positions)
        if start_pos is None:
            print(f"Failed to place agent {agent_id} start position")
            return None
        occupied_positions.add(start_pos)

        # Get goal position (can overlap with other goals but not starts/obstacles)
        goal_occupied = occupied_positions - {s for s in occupied_positions if s in obstacles}
        goal_pos = get_random_position(occupied_positions)
        if goal_pos is None:
            print(f"Failed to place agent {agent_id} goal position")
            return None
        occupied_positions.add(goal_pos)

        input_dict["agents"].append({
            "start": list(start_pos),
            "goal": list(goal_pos),
            "name": f"agent{agent_id}"
        })

    return input_dict


def data_gen(input_dict: Dict, output_path: Path, time_limit: float = None) -> bool:
    """
    Generate solution for given MAPF instance.

    Args:
        input_dict: MAPF instance configuration
        output_path: Path to save solution
        time_limit: seconds to spend on this instance before giving up

    Returns:
        True if solution found, False otherwise
    """
    output_path.mkdir(parents=True, exist_ok=True)

    param = input_dict
    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param["agents"]

    env = Environment(dimension, agents, obstacles)

    # Search for solution
    cbs = CBS(env, verbose=False, time_limit=time_limit)
    solution = cbs.search()

    if not solution:
        return False

    # Write solution file
    output = {
        "schedule": solution,
        "cost": env.compute_solution_cost(solution)
    }

    solution_file = output_path / "solution.yaml"
    with open(solution_file, "w") as f:
        yaml.safe_dump(output, f)

    # Write input parameters file
    parameters_file = output_path / "input.yaml"
    with open(parameters_file, "w") as f:
        yaml.safe_dump(param, f)

    return True


def _solve_case(job):
    """Generate and solve one instance. Top-level so it survives pickling."""
    index, path, config, seed = job
    np.random.seed(seed)

    inpt = gen_input(
        tuple(config["map_shape"]),
        config["nb_obstacles"],
        config["nb_agents"],
    )
    if inpt is None:
        return index, False

    case_path = Path(path) / f"case_{index}"
    if data_gen(inpt, case_path, time_limit=config.get("cbs_time_limit")):
        return index, True

    if case_path.exists():
        shutil.rmtree(case_path)
    return index, False


def create_solutions(path: Path, num_cases: int, config: Dict, workers: int = None):
    """
    Create multiple MAPF instances and their solutions.

    Cases are independent, so they are solved across a process pool. CBS is a
    branch-heavy search that spends its time in the interpreter, which means
    threads would serialise on the GIL but processes scale with cores.

    Args:
        path: Base path for dataset
        num_cases: Number of cases to generate
        config: Configuration dictionary
        workers: Process count; defaults to one per core, 1 disables the pool
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Count existing cases
    existing_cases = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")]
    cases_ready = len(existing_cases)

    print(f"Generating solutions (starting from case {cases_ready})")

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    pending = list(range(cases_ready, num_cases))
    if not pending:
        print("Nothing to generate")
        return

    # Seed per case so a run is reproducible regardless of worker scheduling.
    base_seed = config.get("seed", 0)
    jobs = [(i, str(path), config, base_seed + i) for i in pending]

    successful = 0
    failed = 0
    done = 0

    def report():
        if done % 25 == 0:
            print(
                f"Solution -- [{done}/{len(jobs)}] "
                f"(Success: {successful}, Failed: {failed})",
                flush=True,
            )

    if workers == 1:
        for job in jobs:
            _, ok = _solve_case(job)
            successful += ok
            failed += not ok
            done += 1
            report()
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _, ok in pool.map(_solve_case, jobs, chunksize=4):
                successful += ok
                failed += not ok
                done += 1
                report()

    print(f"Generation complete: {successful} successful, {failed} failed")
    print(f"Cases stored in {path}")


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(description="Generate MAPF training dataset")
    parser.add_argument(
        "--path",
        type=str,
        default="dataset/train",
        help="Output path for dataset"
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=100,
        help="Number of cases to generate"
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of agents"
    )
    parser.add_argument(
        "--obstacles",
        type=int,
        default=5,
        help="Number of obstacles"
    )
    parser.add_argument(
        "--map-size",
        type=int,
        nargs=2,
        default=[10, 10],
        help="Map dimensions (width height)"
    )
    args = parser.parse_args()

    config = {
        "device": "cpu",
        "map_shape": args.map_size,
        "nb_agents": args.agents,
        "nb_obstacles": args.obstacles,
    }

    create_solutions(Path(args.path), args.num_cases, config)


if __name__ == "__main__":
    # Default configuration for testing
    test_config = {
        "device": "cpu",
        "map_shape": [8, 8],
        "nb_agents": 4,
        "nb_obstacles": 5,
    }

    # Use cross-platform path
    test_path = Path("dataset") / "obs_test"

    # Generate small test dataset
    create_solutions(test_path, 2, test_config)