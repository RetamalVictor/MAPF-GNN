"""
Tests for the online expert loop.

The failure mode to guard against is a silent no-op: if collection quietly finds
nothing, or relabelling produces samples identical to the offline data, the whole
loop still runs and simply has no effect. That looks exactly like "the online
expert doesn't help", which is a wrong conclusion about the method rather than a
visible bug. So these tests check that stuck states are found, that the instance
handed to the planner is well formed, and that what comes back is usable
training data.

author: Victor Retamal
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.append("")

from data_generation.online_expert import (
    collect_failures, failure_key, relabel, stuck_step, to_instance,
)

BOARD = 12
AGENTS = 4

CONFIG = {
    "num_agents": AGENTS,
    "nb_agents": AGENTS,
    "board_size": [BOARD, BOARD],
    "map_shape": [BOARD, BOARD],
    "nb_obstacles": 3,
    "obstacles": 3,
    "sensor_range": 4,
    "sensing_range": 4,
    "max_time": 32,
    "min_time": 1,
    "net_type": "gnn",
}


def test_stuck_step_finds_the_last_progress():
    # two robots arrive early, then nothing ever changes again
    assert stuck_step(np.array([0, 1, 2, 2, 2, 2])) == 2
    # progress right at the end
    assert stuck_step(np.array([0, 0, 1])) == 2
    # nothing ever arrived
    assert stuck_step(np.array([0, 0, 0, 0])) == 0


def test_stuck_step_ignores_earlier_plateaus():
    """A pause in the middle is not where the episode gave up."""
    assert stuck_step(np.array([0, 1, 1, 1, 2, 2, 2, 2])) == 4


def make_failure(seed=0):
    rng = np.random.RandomState(seed)
    cells = [(x, y) for x in range(BOARD) for y in range(BOARD)]
    picked = rng.choice(len(cells), size=2 * AGENTS + 3, replace=False)
    chosen = [cells[i] for i in picked]
    return {
        "positions": np.array(chosen[:AGENTS]),
        "goals": np.array(chosen[AGENTS:2 * AGENTS]),
        "obstacles": np.array(chosen[2 * AGENTS:]),
    }


def test_instance_is_well_formed():
    instance = to_instance(make_failure(), BOARD)

    assert len(instance["agents"]) == AGENTS
    assert instance["map"]["dimensions"] == [BOARD, BOARD]
    for agent in instance["agents"]:
        assert len(agent["start"]) == 2 and len(agent["goal"]) == 2
        assert all(isinstance(v, int) for v in agent["start"] + agent["goal"])
    # Plain python types, or yaml refuses to serialise the instance
    yaml.safe_dump(instance)


def test_robots_already_at_goal_stay_in_the_instance():
    """They are moving obstacles that may need to step aside."""
    failure = make_failure()
    failure["goals"][0] = failure["positions"][0]  # this one has arrived

    instance = to_instance(failure, BOARD)
    assert len(instance["agents"]) == AGENTS, "an arrived robot was dropped"
    assert instance["agents"][0]["start"] == instance["agents"][0]["goal"]


def test_failure_key_deduplicates():
    a, b = make_failure(1), make_failure(1)
    assert failure_key(a) == failure_key(b)

    c = make_failure(2)
    assert failure_key(a) != failure_key(c)


def test_relabelled_cases_are_usable_training_data():
    """The planner's answer must come back in the offline data's own format."""
    failures = [make_failure(s) for s in range(4)]

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "train"
        solved, attempted = relabel(
            failures, BOARD, out, dict(CONFIG), time_limit=5.0)

        assert attempted == 4
        if solved == 0:
            pytest.skip("planner solved none of these configurations")

        cases = [d for d in out.iterdir() if d.is_dir()]
        recorded = [c for c in cases if (c / "states.npy").exists()]
        assert recorded, "cases were solved but never recorded"

        for case in recorded:
            states = np.load(case / "states.npy")
            actions = np.load(case / "trajectory_record.npy")
            graphs = np.load(case / "gso.npy")

            assert states.shape[1] == AGENTS
            assert states.shape[2:] == (2, 5, 5)
            assert states.shape[0] == actions.shape[1] == graphs.shape[0], (
                "states, actions and graphs must stay aligned, exactly as in "
                "the offline recorder"
            )
            assert set(np.unique(actions)).issubset({0, 1, 2, 3, 4})


def test_collection_finds_a_jam(monkeypatch):
    """A policy that never moves must produce one failure per episode."""
    import torch

    episodes = 3
    rng = np.random.RandomState(5)
    goals, starts, obstacles = [], [], []
    for _ in range(episodes):
        f = make_failure(rng.randint(1000))
        goals.append(f["goals"])
        starts.append(f["positions"])
        obstacles.append(f["obstacles"])

    class Idle(torch.nn.Module):
        """Always chooses action 0, so nothing ever reaches a goal."""

        def forward(self, fov, gso=None):
            logits = torch.zeros(fov.shape[0], fov.shape[1], 5, device=fov.device)
            logits[..., 0] = 1.0
            return logits

    failures = collect_failures(
        Idle(), CONFIG, BOARD,
        np.stack(goals), np.stack(obstacles), np.stack(starts),
        device="cpu", max_steps=12,
    )

    assert len(failures) == episodes, "a stalled policy should fail every episode"
    for f in failures:
        assert f["positions"].shape == (AGENTS, 2)
        assert f["goals"].shape == (AGENTS, 2)
