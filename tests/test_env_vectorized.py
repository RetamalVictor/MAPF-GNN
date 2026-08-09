"""
Equivalence tests: VectorizedGraphEnv must behave exactly like GraphEnv.

The batched environment exists only to make rollouts fast. If it drifts from
the reference implementation the speed is worthless, so every observation the
policy consumes - the field of view and the communication graph - is compared
step by step against GraphEnv driven with the same actions.

author: Victor Retamal
"""
import numpy as np
import pytest
import torch

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from grid.env_vectorized import VectorizedGraphEnv


CONFIG = {
    "num_agents": 5,
    "board_size": [16, 16],
    "map_shape": [5, 5],
    "obstacles": 6,
    "sensing_range": 4,
    "max_time": 32,
    "min_time": 1,
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def rollout_reference(config, goals, obstacles, starts, actions):
    """Run GraphEnv for one episode, returning per-step observations."""
    env = GraphEnv(
        config,
        goal=goals,
        obstacles=obstacles,
        starting_positions=starts,
        sensing_range=config["sensing_range"],
    )
    obs = env.reset()
    fovs, adjs, positions = [obs["fov"].copy()], [obs["adj_matrix"].copy()], []
    positions.append(env.getPositions().copy())

    emb = env.getEmbedding()
    for step in actions:
        obs, _, _, _ = env.step(step, emb)
        fovs.append(obs["fov"].copy())
        adjs.append(obs["adj_matrix"].copy())
        positions.append(env.getPositions().copy())
    return fovs, adjs, positions


def build_episode(seed):
    rng = np.random.RandomState(seed)
    board = CONFIG["board_size"]
    obstacles = create_obstacles(board, CONFIG["obstacles"])
    goals = create_goals(board, CONFIG["num_agents"], obstacles)

    occupied = {tuple(o) for o in obstacles} | {tuple(g) for g in goals}
    starts = []
    while len(starts) < CONFIG["num_agents"]:
        cell = (rng.randint(board[0]), rng.randint(board[1]))
        if cell not in occupied:
            occupied.add(cell)
            starts.append(cell)
    return obstacles, goals, np.array(starts, dtype=np.int32)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_matches_reference_environment(seed):
    obstacles, goals, starts = build_episode(seed)
    rng = np.random.RandomState(1000 + seed)
    actions = [
        rng.randint(0, 5, size=CONFIG["num_agents"]) for _ in range(24)
    ]

    fovs, adjs, positions = rollout_reference(
        CONFIG, goals, obstacles, starts, actions
    )

    env = VectorizedGraphEnv(
        CONFIG,
        goals=goals[None],
        obstacles=obstacles[None],
        starting_positions=starts[None],
        device=DEVICE,
    )
    obs = env.get_observations()

    assert np.array_equal(
        env.get_positions()[0].cpu().numpy(), positions[0]
    ), "initial positions differ"
    np.testing.assert_allclose(
        obs["fov"][0].cpu().numpy(), fovs[0], atol=1e-6, err_msg="initial FOV differs"
    )
    np.testing.assert_allclose(
        obs["adj_matrix"][0].cpu().numpy(), adjs[0], atol=1e-6,
        err_msg="initial graph differs",
    )

    for t, step in enumerate(actions, start=1):
        obs, _ = env.step(step[None])
        np.testing.assert_array_equal(
            env.get_positions()[0].cpu().numpy(), positions[t],
            err_msg=f"positions diverge at step {t}",
        )
        np.testing.assert_allclose(
            obs["fov"][0].cpu().numpy(), fovs[t], atol=1e-6,
            err_msg=f"FOV diverges at step {t}",
        )
        np.testing.assert_allclose(
            obs["adj_matrix"][0].cpu().numpy(), adjs[t], atol=1e-6,
            err_msg=f"graph diverges at step {t}",
        )


def test_episodes_are_independent():
    """Batched episodes must not leak state into each other."""
    episodes = [build_episode(seed) for seed in (10, 11, 12)]
    goals = np.stack([e[1] for e in episodes])
    starts = np.stack([e[2] for e in episodes])
    obstacles = np.stack([e[0] for e in episodes])

    rng = np.random.RandomState(7)
    actions = [
        rng.randint(0, 5, size=(len(episodes), CONFIG["num_agents"]))
        for _ in range(16)
    ]

    batched = VectorizedGraphEnv(
        CONFIG, goals=goals, obstacles=obstacles,
        starting_positions=starts, device=DEVICE,
    )
    for step in actions:
        batched.step(step)
    batched_positions = batched.get_positions().cpu().numpy()

    for i, (obs_i, goals_i, starts_i) in enumerate(episodes):
        single = VectorizedGraphEnv(
            CONFIG, goals=goals_i[None], obstacles=obs_i[None],
            starting_positions=starts_i[None], device=DEVICE,
        )
        for step in actions:
            single.step(step[i][None])
        np.testing.assert_array_equal(
            single.get_positions()[0].cpu().numpy(), batched_positions[i],
            err_msg=f"episode {i} differs when run batched",
        )


def test_goal_metrics():
    obstacles, goals, starts = build_episode(42)
    env = VectorizedGraphEnv(
        CONFIG, goals=goals[None], obstacles=obstacles[None],
        starting_positions=goals[None], device=DEVICE,
    )
    assert bool(env.all_at_goal()[0]), "agents placed on their goals should count"
    assert float(env.success_rate()[0]) == pytest.approx(1.0)
