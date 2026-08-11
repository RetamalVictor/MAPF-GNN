"""How wide should the communication radius be?

The graph is built by joining each robot to its four nearest neighbours within
`sensing_range` cells. Measured at the trained range of 4, a robot averages
about 1.4 links, so three hops of message passing reach fewer than three other
robots - the graph is a sparse forest, and the network is deeper than the graph
it runs on.

That suggests the range, not the number of hops, is the thing worth changing.
This sweeps it.

The catch is that the range is an *environment* parameter, not a weight. The
policy was trained at 4, so every wider setting hands the graph filter a
neighbourhood denser than anything it saw, and the aggregated message shifts
distribution. More information and worse-conditioned input pull in opposite
directions, so the answer has to be measured rather than argued.

The non-communicating baseline is included as a control: it has no graph, so its
column must stay flat. If it moves, the harness is wrong.

Usage:
    python benchmarks/benchmark_sensing.py --agents 100
    python benchmarks/benchmark_sensing.py --agents 40 100 --episodes 200
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.benchmark_transfer import build_network, rollout

DEFAULT_RANGES = [2, 3, 4, 5, 6, 8, 10, 14]


def matched_board(agents, base_agents=10, base_board=20):
    """Grow the board with the fleet so robots per cell stays fixed."""
    density = base_agents / (base_board ** 2)
    return int(round((agents / density) ** 0.5))


def load(path, device):
    directory = Path(path)
    config = yaml.safe_load((directory / "config.yaml").read_text()) \
        if (directory / "config.yaml").exists() \
        else yaml.safe_load(Path(f"configs/config_{directory.name}.yaml").read_text())
    config["device"] = device

    model = build_network(config)
    model.load_state_dict(torch.load(directory / "best_model.pt", map_location=device))
    model.to(device).eval()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["results/baseline_a10b20", "results/gnn_k3_a10b20"])
    parser.add_argument("--agents", nargs="+", type=int, default=[100])
    parser.add_argument("--ranges", nargs="+", type=int, default=DEFAULT_RANGES)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--horizon-factor", type=int, default=3)
    parser.add_argument("--obstacle-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/sensing_sweep.json")
    args = parser.parse_args()

    loaded = {Path(p).name: load(p, args.device) for p in args.models}
    record = {
        "note": (
            "Success rate against communication radius. The policies were trained "
            "at sensing_range 4; wider settings are out of distribution. Matched "
            "density, horizon 3x board width."
        ),
        "episodes": args.episodes,
        "temperature": args.temperature,
        "trained_range": 4,
        "results": [],
    }

    for agents in args.agents:
        board = matched_board(agents)
        obstacles = max(1, round(args.obstacle_fraction * board * board))
        horizon = args.horizon_factor * board

        print(f"\n{agents} robots on {board}x{board}, {obstacles} obstacles, "
              f"horizon {horizon}, {args.episodes} episodes, T={args.temperature}")
        header = "range  links" + "".join(f"{name:>22}" for name in loaded)
        print(header)

        for sensing in args.ranges:
            row = {"agents": agents, "board": board, "sensing_range": sensing, "alpha": {}}
            line = f"{sensing:5d}"
            links = None

            for name, (model, config) in loaded.items():
                scored = rollout(
                    model,
                    dict(config, sensing_range=sensing),
                    board=board,
                    agents=agents,
                    obstacles=obstacles,
                    episodes=args.episodes,
                    seed=args.seed,
                    device=args.device,
                    max_steps=horizon,
                    temperature=args.temperature,
                )
                row["alpha"][name] = scored["alpha"]
                row["at_goal"] = scored["at_goal"]
                if config["net_type"] != "baseline":
                    links = scored["neighbours"]
                    row["links"] = links
                line += f"{scored['alpha']:>22.3f}"

            print(f"{line[:5]}{(links if links else 0):7.2f}{line[5:]}"
                  + ("   <- as trained" if sensing == 4 else ""))
            record["results"].append(row)

    Path(args.out).write_text(json.dumps(record, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
