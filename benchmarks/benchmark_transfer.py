"""Does a policy trained on a small team scale to a large one?

CBS is only tractable for a handful of agents, so the expert data is generated
with 5. Every weight in these models is shared across agents and every
observation is local, so the same policy can be run on any fleet size. This
script measures how well that holds: train on 5, evaluate on 5, 10, 20, 40...

The communication graph is what should make the difference. With few agents on a
large board the fleet is mostly isolated and the GNN has nothing to aggregate;
as the board fills up, neighbours appear and communication starts to matter.

Usage:
    python benchmarks/benchmark_transfer.py --models results/baseline results/gnn_k3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from grid.env_graph_gridv1 import create_goals, create_obstacles
from grid.env_vectorized import VectorizedGraphEnv
from policy import add_temperature_argument, select_actions


def build_network(config):
    if config["net_type"] == "baseline":
        from models.framework_baseline import Network
    elif config.get("msg_type", "gcn") == "message":
        from models.framework_gnn_message import Network
    else:
        from models.framework_gnn import Network
    return Network(config)


def make_episodes(board, agents, obstacles, count, seed):
    """Independent episodes with distinct starts, goals and obstacle layouts."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    out = []
    for _ in range(count):
        obs = create_obstacles([board, board], obstacles)
        goals = create_goals([board, board], agents, obs)
        blocked = {tuple(o) for o in obs} | {tuple(g) for g in goals}
        starts = []
        while len(starts) < agents:
            cell = (rng.randint(board), rng.randint(board))
            if cell not in blocked:
                blocked.add(cell)
                starts.append(cell)
        out.append((obs, goals, np.array(starts, dtype=np.int32)))
    return (
        np.stack([e[0] for e in out]),
        np.stack([e[1] for e in out]),
        np.stack([e[2] for e in out]),
    )


def rollout(model, config, board, agents, obstacles, episodes, seed, device,
            max_steps, temperature=0.0):
    obs_arr, goals, starts = make_episodes(board, agents, obstacles, episodes, seed)
    env = VectorizedGraphEnv(
        dict(config, board_size=[board, board]),
        goals=goals, obstacles=obs_arr, starting_positions=starts, device=device,
    )
    obs = env.get_observations()
    is_gnn = config["net_type"] != "baseline"

    finished = torch.zeros(episodes, dtype=torch.bool, device=env.device)
    generator = torch.Generator(device=env.device).manual_seed(seed)
    neighbours = []

    for _ in range(max_steps):
        neighbours.append(obs["adj_matrix"].sum(dim=2).mean().item())
        with torch.no_grad():
            logits = model(obs["fov"], obs["adj_matrix"]) if is_gnn else model(obs["fov"])
        actions = select_actions(logits, temperature, generator)
        actions = torch.where(finished.unsqueeze(1), torch.zeros_like(actions), actions)
        obs, done = env.step(actions)
        finished = finished | done
        if bool(finished.all()):
            break

    rates = env.success_rate()
    return {
        "alpha": float((rates == 1.0).float().mean()),
        "complete": float((rates == 1.0).float().mean()),
        "at_goal": float(rates.mean()),
        "neighbours": float(np.mean(neighbours)),
    }


def scenario(args, agents):
    """Board size and obstacle count for a given fleet size.

    Fixed board: adding agents also crowds them, which stresses the policy but
    conflates team size with density. Matched density: the board grows as
    sqrt(agents) so agents per cell, and obstacle coverage, stay put.
    """
    if not args.match_density:
        return args.board, args.obstacles

    base_agents = args.agents[0]
    density = base_agents / (args.board ** 2)
    board = max(args.board, int(round((agents / density) ** 0.5)))
    coverage = args.obstacles / (args.board ** 2)
    return board, max(1, int(round(coverage * board ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True,
                        help="result directories holding best_model.pt and config.yaml")
    parser.add_argument("--agents", nargs="+", type=int,
                        default=[5, 10, 20, 30, 40, 60])
    parser.add_argument("--board", type=int, default=28)
    parser.add_argument("--obstacles", type=int, default=8)
    parser.add_argument("--match-density", action="store_true",
                        help="Grow the board with the fleet so agents per cell "
                             "stays at the first point's value. Without this, "
                             "adding agents also crowds them, and the two "
                             "effects cannot be told apart.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Fixed episode horizon for every point")
    parser.add_argument("--horizon-factor", type=float, default=None,
                        help="Episode horizon as a multiple of the board width. "
                             "A fixed horizon is unfair once the board grows with "
                             "the fleet, and an over-generous one lets a sampling "
                             "policy brute-force its way to the goal, which hides "
                             "the differences between architectures.")
    parser.add_argument("--seed", type=int, default=42)
    add_temperature_argument(parser)
    parser.add_argument("--save", default=None)
    args = parser.parse_args()

    from evaluate import _load_yaml

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = {}

    for model_dir in args.models:
        model_dir = Path(model_dir)
        with open(model_dir / "config.yaml") as f:
            config = _load_yaml(f)
        config["device"] = device
        model = build_network(config).to(device).eval()
        model.load_state_dict(
            torch.load(model_dir / "best_model.pt", map_location=device)
        )
        default_steps = args.max_steps or config["max_steps"] + 10

        results[model_dir.name] = {}
        for agents in args.agents:
            board, obstacles = scenario(args, agents)
            max_steps = (int(args.horizon_factor * board)
                         if args.horizon_factor else default_steps)
            results[model_dir.name][agents] = rollout(
                model, config, board, agents, obstacles,
                args.episodes, args.seed, device, max_steps, args.temperature,
            )
            results[model_dir.name][agents]["board"] = board

    trained_on = args.agents[0]
    print(f"\nTrained on {trained_on} agents, board {args.board}x{args.board}, "
          f"{args.obstacles} obstacles, {args.episodes} episodes per point\n")

    header = f"{'agents':<9}{'board':<8}{'neigh':<9}" + "".join(f"{name:<22}" for name in results)
    print(header)
    print(f"{'':<9}{'':<8}{'':<9}" + "".join(f"{'complete':<11}{'at goal':<11}" for _ in results))
    print("-" * len(header))
    for agents in args.agents:
        first = next(iter(results))
        row = f"{agents:<9}{results[first][agents]['board']:<8}{results[first][agents]['neighbours']:<9.2f}"
        for name in results:
            r = results[name][agents]
            row += f"{r['complete']:<11.3f}{r['at_goal']:<11.3f}"
        print(row)

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"\nsaved to {args.save}")


if __name__ == "__main__":
    main()
