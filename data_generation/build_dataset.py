"""Build a training set spanning several obstacle densities.

The original set is a single density - eight obstacles on a 20x20 board, two
percent - so the policy only ever saw nearly empty maps. The browser demo has a
density slider, and a policy that has seen one density is being asked to
extrapolate the moment anyone drags it.

CBS is comfortable across the whole range, and in fact gets *faster* as
obstacles are added, because they prune the search: measured at ten agents on
20x20 with a ten second limit, 0.27 s/case at 2% against 0.02 s/case at 15%,
solving 37-40 of every 40 throughout. So there is no reason to train on one.

Each density becomes its own dataset directory; the training config lists them
all and the loader concatenates them.

Recording is separate from solving, and much cheaper: the CBS solution does not
depend on the sensing range or the neighbour cap, only the observations do. To
re-record an existing set under different graph settings, pass --record-only.

Usage:
    python data_generation/build_dataset.py --root dataset/dense_r8 \
        --obstacles 8 20 40 60 --cases 500 --sensing-range 8 --max-neighbours 0
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_generation.dataset_gen import create_solutions
from data_generation.record import record_env
from data_generation.trajectory_parser import parse_dataset_trajectories


def build(root, agents, board, obstacles, cases, sensing_range, max_neighbours,
          time_limit, record_only, workers):
    """Generate and record one dataset per obstacle count, returning their paths."""
    paths = []

    for count in obstacles:
        # Named {agents}_{obstacles}_{board}, the convention tests/test_models.py
        # parses to check a config against the data it was trained on.
        path = Path(root) / f"{agents}_{count}_{board}" / "train"
        print(f"\n=== {count} obstacles on {board}x{board} "
              f"({100 * count / (board * board):.1f}% of the board) ===")

        solve_config = {
            "device": "cpu",
            "map_shape": [board, board],
            "nb_agents": agents,
            "nb_obstacles": count,
            "cbs_time_limit": time_limit,
        }

        if not record_only:
            started = time.time()
            create_solutions(path, cases, solve_config, workers=workers)
            print(f"  solved in {time.time() - started:.0f}s")
            parse_dataset_trajectories(str(path))

        # The recorder reads sensor_range; training configs say sensing_range.
        # They must carry the same number or the graph the policy trains
        # against is not the graph it is evaluated against.
        record_config = {
            "num_agents": agents,
            "nb_agents": agents,
            "board_size": [board, board],
            "map_shape": [board, board],
            "sensor_range": sensing_range,
            "sensing_range": sensing_range,
            # 0 on the command line means "no cap": aggregate over everything
            # in range. GraphEnv reads None as uncapped.
            "max_neighbours": max_neighbours or None,
            "max_time": 32,
            "min_time": 1,
        }

        started = time.time()
        record_env(str(path), record_config)
        print(f"  recorded in {time.time() - started:.0f}s")
        paths.append(str(path).replace("\\", "/"))

    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="parent directory for the datasets")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--board", type=int, default=20)
    parser.add_argument("--obstacles", nargs="+", type=int, default=[8, 20, 40, 60])
    parser.add_argument("--cases", type=int, default=500)
    parser.add_argument("--sensing-range", type=int, default=8)
    parser.add_argument("--max-neighbours", type=int, default=0,
                        help="0 for uncapped; 4 reproduces the original graph")
    parser.add_argument("--cbs-time-limit", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--record-only", action="store_true",
                        help="re-record existing solutions under new graph settings")
    args = parser.parse_args()

    paths = build(
        args.root, args.agents, args.board, args.obstacles, args.cases,
        args.sensing_range, args.max_neighbours, args.cbs_time_limit,
        args.record_only, args.workers,
    )

    print("\nAdd to the training config:\n")
    print("train:")
    print("  root_dir:")
    for path in paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
