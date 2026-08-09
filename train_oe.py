"""Train with an online expert.

Each round: train the policy, run it, find the configurations it jams in, ask CBS
what to do from there, and fold those answers back into the dataset. The expert is
queried from states the policy actually reached, which is the part plain
behaviour cloning cannot provide - it only ever sees states an optimal planner
visits, and a deadlock is by definition not one of those.

Reports success rate alpha - the fraction of cases where *every* robot reaches
its goal - because that is the metric the literature quotes, and it is the one
that shows the problem.

Usage:
    python train_oe.py --config configs/config_gnn_msg_k3_a10b20.yaml --rounds 3
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from benchmarks.benchmark_transfer import build_network, make_episodes
from data_generation.online_expert import collect_failures, failure_key, relabel
from evaluate import _load_yaml
from grid.env_vectorized import VectorizedGraphEnv

# Fleet sizes to report alpha at each round. The training fleet first, then
# sizes the policy has never seen.
EVAL_POINTS = [(10, 20), (20, 28), (40, 40), (60, 49)]
EVAL_EPISODES = 200
EVAL_SEED = 42


def evaluate_alpha(model, config, device, episodes=EVAL_EPISODES):
    """Success rate alpha, plus the softer fraction-of-robots metric."""
    results = {}
    is_gnn = config["net_type"] != "baseline"
    horizon = 300

    for agents, board in EVAL_POINTS:
        obstacles = max(1, round(8 / (28 ** 2) * board ** 2))
        obs_arr, goals, starts = make_episodes(board, agents, obstacles, episodes, EVAL_SEED)
        env = VectorizedGraphEnv(
            dict(config, board_size=[board, board]),
            goals=goals, obstacles=obs_arr, starting_positions=starts, device=device,
        )
        obs = env.get_observations()
        finished = torch.zeros(episodes, dtype=torch.bool, device=env.device)

        for _ in range(horizon):
            with torch.no_grad():
                logits = model(obs["fov"], obs["adj_matrix"]) if is_gnn else model(obs["fov"])
            actions = logits.argmax(dim=2)
            actions = torch.where(finished.unsqueeze(1), torch.zeros_like(actions), actions)
            obs, done = env.step(actions)
            finished = finished | done
            if bool(finished.all()):
                break

        rates = env.success_rate()
        results[agents] = {
            "alpha": float((rates == 1.0).float().mean()),
            "at_goal": float(rates.mean()),
        }
    return results


def load_policy(exp_name, config, device):
    model = build_network(config).to(device).eval()
    checkpoint = Path("results") / exp_name / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


def gather(model, config, device, episodes, agents, board, obstacles, seed, seen):
    """Roll the policy out and return the stuck states we have not seen before."""
    obs_arr, goals, starts = make_episodes(board, agents, obstacles, episodes, seed)
    failures = collect_failures(
        model, config, board, goals, obs_arr, starts, device,
        max_steps=config["max_steps"] + 10,
    )
    fresh = []
    for failure in failures:
        key = failure_key(failure)
        if key in seen:
            continue
        seen.add(key)
        fresh.append(failure)
    return failures, fresh


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=300,
                   help="Rollout episodes per round when hunting for failures")
    p.add_argument("--collect-agents", type=int, default=None,
                   help="Fleet size to collect failures at. Defaults to the "
                        "training fleet size, and must match it: a sample's "
                        "tensors are shaped by the agent count, so relabelled "
                        "cases with a different fleet size cannot be batched "
                        "alongside the offline ones.")
    p.add_argument("--collect-board", type=int, default=None,
                   help="Board to collect on. Defaults to the training board.")
    p.add_argument("--oe-share", type=float, default=0.33,
                   help="Ceiling on the online expert's share of the dataset")
    p.add_argument("--time-limit", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_config = _load_yaml(open(args.config))
    exp_name = base_config["exp_name"]
    oe_name = f"{exp_name}_oe"

    train_agents = base_config["num_agents"]
    if args.collect_agents is None:
        args.collect_agents = train_agents
    if args.collect_board is None:
        args.collect_board = base_config["board_size"][0]
    if args.collect_agents != train_agents:
        raise SystemExit(
            f"--collect-agents {args.collect_agents} does not match the training "
            f"fleet size {train_agents}. Samples are shaped (agents, ...), so "
            f"relabelled cases have to carry the same fleet size to be batched "
            f"with the offline ones. Collecting where the policy jams hardest "
            f"would need the loader to bucket batches by fleet size."
        )

    offline_root = base_config["train"]["root_dir"]
    offline_cases = len(list((Path(offline_root) / "train").iterdir()))
    oe_root = Path("dataset") / f"{oe_name}_data"
    (oe_root / "train").mkdir(parents=True, exist_ok=True)

    # Cap the online expert's share: at share s of the total, it contributes
    # offline * s / (1 - s) cases. Too much recovery data and the policy gets
    # worse at ordinary progress.
    oe_cap = int(offline_cases * args.oe_share / (1 - args.oe_share))

    config_path = Path("configs") / f"{oe_name}.yaml"
    history = []
    seen = set()
    collected = 0

    print(f"device {device} | offline cases {offline_cases} | oe cap {oe_cap}", flush=True)

    for round_i in range(args.rounds + 1):
        # Round 0 trains on offline data alone, so its numbers are the baseline
        # every later round is compared against.
        round_config = dict(base_config)
        round_config["exp_name"] = oe_name
        round_config["train"] = dict(base_config["train"])
        if round_i > 0:
            round_config["train"]["root_dir"] = [offline_root, str(oe_root)]
            round_config["train"]["case_caps"] = [None, oe_cap]

        with open(config_path, "w") as f:
            yaml.safe_dump(round_config, f, sort_keys=False)

        print(f"\n===== round {round_i} — training =====", flush=True)
        started = time.time()
        run = subprocess.run(
            [sys.executable, "train.py", "--config", str(config_path),
             "--seed", str(args.seed)],
            capture_output=True, text=True,
        )
        if run.returncode != 0:
            print(run.stdout[-3000:])
            print(run.stderr[-3000:])
            raise SystemExit(f"training failed in round {round_i}")
        train_seconds = time.time() - started

        model = load_policy(oe_name, round_config, device)
        alpha = evaluate_alpha(model, round_config, device)

        entry = {
            "round": round_i,
            "train_seconds": round(train_seconds, 1),
            "oe_cases_available": collected,
            "oe_cases_used": min(collected, oe_cap) if round_i else 0,
            "alpha": {a: round(v["alpha"], 4) for a, v in alpha.items()},
            "at_goal": {a: round(v["at_goal"], 4) for a, v in alpha.items()},
        }
        history.append(entry)
        print(f"round {round_i}: " + "  ".join(
            f"a{a}={v['alpha']:.3f}" for a, v in alpha.items()), flush=True)

        Path("results/oe_history.json").write_text(json.dumps(history, indent=2))
        if round_i == args.rounds:
            break

        print(f"===== round {round_i} — collecting =====", flush=True)
        started = time.time()
        obstacles = max(1, round(8 / (28 ** 2) * args.collect_board ** 2))
        failures, fresh = gather(
            model, round_config, device, args.episodes,
            args.collect_agents, args.collect_board, obstacles,
            args.seed + round_i, seen,
        )
        solved, attempted = relabel(
            fresh, args.collect_board, oe_root / "train", round_config,
            time_limit=args.time_limit, start_index=collected,
        )
        collected += solved
        print(f"  failures {len(failures)}/{args.episodes} | new {len(fresh)} | "
              f"solved {solved}/{attempted} | pool {collected} | "
              f"{time.time() - started:.0f}s", flush=True)

        if solved == 0:
            print("  no new demonstrations - stopping", flush=True)
            break

    print("\n===== summary =====")
    header = f"{'round':<8}{'oe cases':<11}" + "".join(f"a{a:<9}" for a, _ in EVAL_POINTS)
    print(header)
    for entry in history:
        row = f"{entry['round']:<8}{entry['oe_cases_used']:<11}"
        row += "".join(f"{entry['alpha'][a]:<10.3f}" for a, _ in EVAL_POINTS)
        print(row)
    print("\nsaved results/oe_history.json")


if __name__ == "__main__":
    main()
