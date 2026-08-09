"""
Tests for the recording stage of the data pipeline.

Recording replays each CBS solution through the environment and stores the
observation the expert acted on at every step. If the stored observation and
the stored action drift apart by even one timestep, imitation learning quietly
trains the policy to predict the action that produced the current state instead
of the action to take from it. Nothing downstream raises; the model just
underperforms. These tests pin the alignment.

author: Victor Retamal
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import yaml

sys.path.append("")

from data_generation.dataset_gen import create_solutions
from data_generation.trajectory_parser import parse_dataset_trajectories
from data_generation.record import make_env, record_env


CONFIG = {
    "num_agents": 3,
    "nb_agents": 3,
    "map_shape": [12, 12],
    "board_size": [12, 12],
    "nb_obstacles": 4,
    "sensor_range": 4,
    "max_time": 24,
    "min_time": 4,
}


@pytest.fixture(scope="module")
def recorded_dataset():
    """Generate, parse and record a small dataset once for all tests."""
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "train")
    config = dict(CONFIG, path=path)

    create_solutions(path, 6, config)
    parse_dataset_trajectories(path)
    record_env(path, config)

    cases = [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if name.startswith("case_")
        and os.path.exists(os.path.join(path, name, "states.npy"))
    ]
    if not cases:
        pytest.skip("No cases were solved for this seed")
    yield config, cases


def schedule_positions(case_dir):
    """Agent positions per timestep, straight from the CBS solution."""
    with open(os.path.join(case_dir, "solution.yaml")) as f:
        schedule = yaml.safe_load(f)["schedule"]
    agents = sorted(schedule.keys())
    horizon = max(len(schedule[a]) for a in agents)
    positions = np.zeros((horizon, len(agents), 2), dtype=int)
    for j, agent in enumerate(agents):
        path = schedule[agent]
        for t in range(horizon):
            step = path[min(t, len(path) - 1)]
            positions[t, j] = [step["x"], step["y"]]
    return positions


def test_recorded_actions_reproduce_the_expert_path(recorded_dataset):
    """Replaying the stored actions must follow the CBS solution exactly."""
    config, cases = recorded_dataset

    for case_dir in cases:
        expected = schedule_positions(case_dir)
        actions = np.load(os.path.join(case_dir, "trajectory_record.npy"))

        env = make_env(case_dir, config)
        env.reset()
        assert np.array_equal(env.getPositions(), expected[0]), (
            f"{os.path.basename(case_dir)}: environment does not start at the "
            f"first scheduled positions"
        )

        emb = np.ones(env.nb_agents)
        for t in range(actions.shape[1] - 1):
            env.step(actions[:, t], emb)
            target = expected[min(t + 1, len(expected) - 1)]
            assert np.array_equal(env.getPositions(), target), (
                f"{os.path.basename(case_dir)}: diverged from the expert path at "
                f"step {t + 1}. The stored actions are offset from the states."
            )


def test_states_and_actions_have_matching_lengths(recorded_dataset):
    _, cases = recorded_dataset

    for case_dir in cases:
        states = np.load(os.path.join(case_dir, "states.npy"))
        actions = np.load(os.path.join(case_dir, "trajectory_record.npy"))
        graphs = np.load(os.path.join(case_dir, "gso.npy"))

        assert states.shape[0] == actions.shape[1] == graphs.shape[0], (
            f"{os.path.basename(case_dir)}: states {states.shape[0]}, actions "
            f"{actions.shape[1]}, graphs {graphs.shape[0]}"
        )


def test_recorded_graph_matches_the_state_it_is_stored_with(recorded_dataset):
    """gso[t] must be the graph the agents saw when choosing actions[:, t]."""
    config, cases = recorded_dataset

    for case_dir in cases:
        actions = np.load(os.path.join(case_dir, "trajectory_record.npy"))
        graphs = np.load(os.path.join(case_dir, "gso.npy"))[:, 0, :, :]

        env = make_env(case_dir, config)
        obs = env.reset()
        emb = np.ones(env.nb_agents)
        for t in range(actions.shape[1]):
            assert np.array_equal(obs["adj_matrix"], graphs[t]), (
                f"{os.path.basename(case_dir)}: recorded graph at step {t} does "
                f"not match the graph the environment produced"
            )
            obs, _, _, _ = env.step(actions[:, t], emb)


def test_full_trajectories_are_recorded(recorded_dataset):
    """Recording must cover the whole episode, not just the opening moves."""
    _, cases = recorded_dataset

    for case_dir in cases:
        expected = schedule_positions(case_dir)
        states = np.load(os.path.join(case_dir, "states.npy"))
        assert states.shape[0] >= len(expected) - 1, (
            f"{os.path.basename(case_dir)}: recorded {states.shape[0]} steps for "
            f"a {len(expected)}-step solution"
        )
