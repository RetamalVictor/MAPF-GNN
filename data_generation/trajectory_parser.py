"""
Trajectory parser for MAPF solutions.
Converts CBS solution schedules to action sequences for training.

author: Victor Retamal
"""
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def get_longest_path(schedule):
    """Get the longest path length from all agents."""
    longest = 0
    for agent in schedule.keys():
        if len(schedule[agent]) > longest:
            longest = len(schedule[agent])
    return longest


def parse_trajectories(schedule):
    """
    Parse agent trajectories into action sequences.

    Args:
        schedule: Dictionary mapping agent names to trajectory lists

    Returns:
        trajectory: Array of shape (num_agents, max_time) with actions
        startings: Array of shape (num_agents, 2) with starting positions
    """
    longest = get_longest_path(schedule)
    trajectory = np.zeros((len(schedule), longest), dtype=np.int32)

    # Map direction vectors to action indices
    action_map = {
        str((0, 0)): 0,   # Wait
        str((1, 0)): 1,   # Right
        str((0, 1)): 2,   # Up
        str((-1, 0)): 3,  # Left
        str((0, -1)): 4,  # Down
    }

    startings = np.zeros((len(schedule), 2), dtype=np.int32)

    for j, agent in enumerate(schedule.keys()):
        agent_path = schedule[agent]
        startings[j][0] = agent_path[0]["x"]
        startings[j][1] = agent_path[0]["y"]

        # For each time step, compute the action from current to next position
        for i in range(longest):
            if i < len(agent_path) - 1:
                # Action from position i to position i+1
                curr_x, curr_y = agent_path[i]["x"], agent_path[i]["y"]
                next_x, next_y = agent_path[i+1]["x"], agent_path[i+1]["y"]
                trajectory[j][i] = action_map[str((next_x - curr_x, next_y - curr_y))]
            else:
                # No more positions, agent waits
                trajectory[j][i] = 0  # Wait

    return trajectory, np.array(startings)


def parse_dataset_trajectories(path):
    """
    Parse trajectories for all cases in a dataset directory.

    Args:
        path: Path to dataset directory containing case folders
    """
    path = Path(path)
    cases = sorted([d for d in path.iterdir() if d.is_dir() and d.name.startswith("case_")])

    print(f"Parsing trajectories for {len(cases)} cases")

    for idx, case_dir in enumerate(cases):
        solution_file = case_dir / "solution.yaml"

        if not solution_file.exists():
            print(f"Warning: Solution file not found for {case_dir.name}")
            continue

        with open(solution_file) as f:
            schedule = yaml.load(f, Loader=yaml.FullLoader)

        combined_schedule = {}
        combined_schedule.update(schedule["schedule"])

        trajectory, startings = parse_trajectories(combined_schedule)

        # Save trajectory
        trajectory_file = case_dir / "trajectory.npy"
        np.save(trajectory_file, trajectory)

        # Save starting positions
        startings_file = case_dir / "startings.npy"
        np.save(startings_file, startings)

        if idx % 25 == 0:
            print(f"Trajectory -- [{idx}/{len(cases)}]")

    print(f"Trajectory -- [{len(cases)}/{len(cases)}] Complete!")


def main():
    """Main entry point for trajectory parsing."""
    parser = argparse.ArgumentParser(description="Parse MAPF solution trajectories")
    parser.add_argument(
        "path",
        type=str,
        help="Path to dataset directory containing case folders"
    )
    parser.add_argument(
        "--single",
        type=str,
        help="Parse a single solution file instead of dataset"
    )
    args = parser.parse_args()

    if args.single:
        # Parse single file
        with open(args.single) as f:
            schedule = yaml.load(f, Loader=yaml.FullLoader)

        combined_schedule = {}
        combined_schedule.update(schedule["schedule"])

        trajectory, startings = parse_trajectories(combined_schedule)
        print("Trajectory shape:", trajectory.shape)
        print("Starting positions:", startings)
    else:
        # Parse dataset
        parse_dataset_trajectories(args.path)


if __name__ == "__main__":
    # Default path for testing - use command line args in production
    test_path = Path("dataset") / "obs_test"

    if test_path.exists():
        parse_dataset_trajectories(test_path)
    else:
        print(f"Test path {test_path} does not exist. Use command line arguments.")
        main()