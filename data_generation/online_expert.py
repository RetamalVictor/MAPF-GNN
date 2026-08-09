"""Online expert: ask the planner what to do from states the policy got stuck in.

Behaviour cloning fits the policy to the expert's state distribution, but at
deployment the policy visits its own. The first small error lands it somewhere
CBS would never have gone, where there is no training signal, so errors compound
rather than correct. Here that shows up as a cluster of robots that jams and
never clears - and we know these are genuine deadlocks rather than slow episodes,
because extending the evaluation horizon barely moves the numbers.

This module rolls the current policy out, finds the configurations where it
stopped making progress, and relabels those with CBS. The expert is queried from
states the policy actually reached, which is the only part of this that teaches
recovery.

author: Victor Retamal
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from data_generation.dataset_gen import data_gen
from data_generation.record import record_env
from data_generation.trajectory_parser import parse_dataset_trajectories
from grid.env_vectorized import VectorizedGraphEnv


def rollout_with_history(model, config, board, goals, obstacles, starts,
                         device, max_steps):
    """Run the policy and keep the positions and goal counts at every step."""
    env = VectorizedGraphEnv(
        dict(config, board_size=[board, board]),
        goals=goals, obstacles=obstacles, starting_positions=starts, device=device,
    )
    obs = env.get_observations()
    is_gnn = config["net_type"] != "baseline"

    positions = [env.get_positions().cpu().numpy()]
    arrived = [env.at_goal().sum(dim=1).cpu().numpy()]
    finished = torch.zeros(len(goals), dtype=torch.bool, device=env.device)

    for _ in range(max_steps):
        with torch.no_grad():
            logits = model(obs["fov"], obs["adj_matrix"]) if is_gnn else model(obs["fov"])
        actions = logits.argmax(dim=2)
        actions = torch.where(finished.unsqueeze(1), torch.zeros_like(actions), actions)

        obs, done = env.step(actions)
        finished = finished | done
        positions.append(env.get_positions().cpu().numpy())
        arrived.append(env.at_goal().sum(dim=1).cpu().numpy())
        if bool(finished.all()):
            break

    return np.stack(positions), np.stack(arrived), env.at_goal().cpu().numpy()


def stuck_step(arrived_over_time):
    """The step after which no further robot ever reaches its goal.

    Taking the final frame instead would relabel a configuration that has already
    tangled itself further; the last moment progress happened is closer to the
    decision that caused the jam.
    """
    increases = np.flatnonzero(np.diff(arrived_over_time) > 0)
    return int(increases[-1] + 1) if len(increases) else 0


def collect_failures(model, config, board, goals, obstacles, starts,
                     device, max_steps):
    """Configurations where the policy stopped making progress.

    Returns one record per failed episode, holding the positions at the moment
    progress stopped, the original goals, and the obstacles.
    """
    positions, arrived, at_goal = rollout_with_history(
        model, config, board, goals, obstacles, starts, device, max_steps)

    failures = []
    for e in range(len(goals)):
        if at_goal[e].all():
            continue
        t = stuck_step(arrived[:, e])
        failures.append({
            "episode": e,
            "step": t,
            "positions": positions[t, e].copy(),
            "goals": np.asarray(goals[e]).copy(),
            "obstacles": np.asarray(obstacles[e]).copy(),
        })
    return failures


def failure_key(failure):
    """Identity of a stuck configuration, for deduplication.

    Different episodes jam in the same way often enough that without this one
    failure mode floods the dataset.
    """
    return (
        tuple(map(tuple, failure["positions"].tolist())),
        tuple(map(tuple, failure["goals"].tolist())),
    )


def to_instance(failure, board):
    """Turn a stuck configuration into a CBS instance.

    Robots already standing on their goal stay in the instance: they are moving
    obstacles that may have to step aside, and dropping them yields solutions
    that walk straight through them.
    """
    return {
        "agents": [
            {"start": list(map(int, p)), "goal": list(map(int, g)), "name": f"agent{i}"}
            for i, (p, g) in enumerate(zip(failure["positions"], failure["goals"]))
        ],
        "map": {
            "dimensions": [board, board],
            "obstacles": [tuple(map(int, o)) for o in failure["obstacles"]],
        },
    }


def recorder_config(config, board, agents):
    """Config in the shape the offline recorder expects.

    Training configs say sensing_range and the recorder reads sensor_range; they
    must carry the same value, or the recorded communication graph would differ
    from the one the policy was trained against.
    """
    sensing = config.get("sensor_range", config.get("sensing_range", 6))
    return {
        **config,
        "num_agents": agents,
        "nb_agents": agents,
        "board_size": [board, board],
        "map_shape": [board, board],
        "sensor_range": sensing,
        "sensing_range": sensing,
        "max_time": config.get("max_time", 32),
        "min_time": config.get("min_time", 1),
    }


def relabel(failures, board, out_dir, config, time_limit=2.0, start_index=0):
    """Solve each stuck configuration and record it as training data.

    Returns (solved, attempted). The recorded cases land in out_dir in exactly
    the layout the offline generator produces, so the data loader needs no
    special case for them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solved = 0
    for i, failure in enumerate(failures):
        case_path = out_dir / f"case_{start_index + i}"
        if data_gen(to_instance(failure, board), case_path, time_limit=time_limit):
            solved += 1

    if solved:
        agents = len(failures[0]["positions"])
        parse_dataset_trajectories(str(out_dir))
        record_env(str(out_dir), recorder_config(config, board, agents))
    return solved, len(failures)
