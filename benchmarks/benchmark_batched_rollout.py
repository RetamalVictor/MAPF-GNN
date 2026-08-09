"""Sequential vs batched rollouts on identical episodes.

evaluate.py runs one episode at a time, so every step is a batch-1 forward pass.
A model this small costs the same at batch 1 as at batch 128 - the time is
kernel launch latency, not arithmetic - so running the episodes concurrently
should cost roughly the same wall clock as running one of them.

Both paths are driven with the same obstacles, goals and starting positions, and
the script checks they produce identical results before reporting the speedup.

Usage:
    python benchmarks/benchmark_batched_rollout.py --model results/gnn_k3/best_model.pt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from grid.env_vectorized import VectorizedGraphEnv


def load_config(path):
    from evaluate import _load_yaml

    with open(path) as f:
        return _load_yaml(f)


def build_network(config):
    if config["net_type"] == "baseline":
        from models.framework_baseline import Network
    elif config.get("msg_type", "gcn") == "message":
        from models.framework_gnn_message import Network
    else:
        from models.framework_gnn import Network
    return Network(config)


def make_episodes(config, count, seed):
    """Pre-generate episodes so both paths see exactly the same problems."""
    np.random.seed(seed)
    board = config["board_size"]
    episodes = []
    for _ in range(count):
        obstacles = create_obstacles(board, config["obstacles"])
        goals = create_goals(board, config["num_agents"], obstacles)
        blocked = {tuple(o) for o in obstacles} | {tuple(g) for g in goals}
        starts = []
        while len(starts) < config["num_agents"]:
            cell = (np.random.randint(board[0]), np.random.randint(board[1]))
            if cell not in blocked:
                blocked.add(cell)
                starts.append(cell)
        episodes.append((obstacles, goals, np.array(starts, dtype=np.int32)))
    return episodes


def run_sequential(model, config, episodes, device, max_steps):
    complete, at_goal, steps = 0, [], []
    is_gnn = config["net_type"] != "baseline"

    for obstacles, goals, starts in episodes:
        env = GraphEnv(
            config, goal=goals, obstacles=obstacles,
            starting_positions=starts.copy(),
            sensing_range=config.get("sensing_range", 6),
        )
        obs = env.reset()
        emb = env.getEmbedding()
        used = max_steps
        for step in range(max_steps):
            fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(device)
            gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(fov, gso) if is_gnn else model(fov)
            action = logits.cpu().squeeze(0).numpy().argmax(axis=1)
            obs, _, done, _ = env.step(action, emb)
            if done:
                used = step + 1
                break
        rate = env.computeMetrics()[0]
        at_goal.append(rate)
        steps.append(used)
        complete += int(rate == 1.0)
    return complete, float(np.mean(at_goal)), float(np.mean(steps))


def run_batched(model, config, episodes, device, max_steps):
    obstacles = np.stack([e[0] for e in episodes])
    goals = np.stack([e[1] for e in episodes])
    starts = np.stack([e[2] for e in episodes])
    is_gnn = config["net_type"] != "baseline"

    env = VectorizedGraphEnv(
        config, goals=goals, obstacles=obstacles,
        starting_positions=starts, device=device,
    )
    obs = env.get_observations()

    n = len(episodes)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    steps = torch.full((n,), max_steps, dtype=torch.long, device=env.device)

    for step in range(max_steps):
        with torch.no_grad():
            logits = model(obs["fov"], obs["adj_matrix"]) if is_gnn else model(obs["fov"])
        actions = logits.argmax(dim=2)
        # Finished episodes idle in place so their metrics stop changing, which
        # is what breaking out of the sequential loop does.
        actions = torch.where(finished.unsqueeze(1), torch.zeros_like(actions), actions)

        obs, done = env.step(actions)
        newly_done = done & ~finished
        steps = torch.where(newly_done, torch.full_like(steps, step + 1), steps)
        finished = finished | done
        if bool(finished.all()):
            break

    rates = env.success_rate()
    return (
        int((rates == 1.0).sum()),
        float(rates.mean()),
        float(steps.float().mean()),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = Path(args.model)
    config_path = args.config or model_path.parent / "config.yaml"
    config = load_config(config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = device

    model = build_network(config).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))

    max_steps = config["max_steps"] + 10
    episodes = make_episodes(config, args.episodes, args.seed)

    # Warm up so CUDA context creation is not charged to the first path.
    run_batched(model, config, episodes[:2], device, max_steps)

    t0 = time.perf_counter()
    seq = run_sequential(model, config, episodes, device, max_steps)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    bat = run_batched(model, config, episodes, device, max_steps)
    t_bat = time.perf_counter() - t0

    print(f"\nEpisodes: {args.episodes}   device: {device}   model: {model_path}")
    print(f"{'':<12}{'complete':<12}{'at goal':<12}{'steps':<10}{'time':<10}")
    print(f"{'sequential':<12}{seq[0]:<12}{seq[1]:<12.4f}{seq[2]:<10.2f}{t_seq:<10.2f}s")
    print(f"{'batched':<12}{bat[0]:<12}{bat[1]:<12.4f}{bat[2]:<10.2f}{t_bat:<10.2f}s")
    print(f"\nspeedup: {t_seq / t_bat:.1f}x")

    if seq[0] != bat[0] or abs(seq[1] - bat[1]) > 1e-6:
        print("\nWARNING: results differ between the two paths")
        return 1
    print("results identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
