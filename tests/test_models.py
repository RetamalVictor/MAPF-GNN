"""
Tests for the model frameworks and the shipped configuration files.

These exist because a broken import inside models/ once made train.py fail on
startup for every configuration while the rest of the suite stayed green: no
test imported the model code, so nothing noticed.
"""

import warnings
from pathlib import Path

import pytest
import torch
import yaml

CONFIG_DIR = Path(__file__).parent.parent / "configs"
CONFIG_FILES = sorted(CONFIG_DIR.glob("config_*.yaml"))


def load_config(path):
    # Reuse the loader evaluate.py ships with: older checkpoints stored a
    # torch.device object in their config, which plain safe_load rejects.
    from evaluate import _load_yaml

    with open(path) as f:
        config = _load_yaml(f)
    config["device"] = torch.device("cpu")
    return config


def build_network(config):
    """Resolve and instantiate the network the way train.py and evaluate.py do."""
    net_type = config["net_type"]
    if net_type == "baseline":
        from models.framework_baseline import Network
    elif net_type == "gnn":
        if config.get("msg_type", "gcn") == "message":
            from models.framework_gnn_message import Network
        else:
            from models.framework_gnn import Network
    else:
        raise ValueError(f"Unknown net_type: {net_type}")
    return Network(config)


def test_every_framework_imports():
    """Every model module must import; train.py imports these at startup."""
    from models.framework_baseline import Network as Baseline
    from models.framework_gnn import Network as GNN
    from models.framework_gnn_message import Network as GNNMessage

    assert Baseline is not None
    assert GNN is not None
    assert GNNMessage is not None


def test_config_files_exist():
    assert CONFIG_FILES, f"No config files found in {CONFIG_DIR}"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_builds_and_runs_forward(config_path):
    """Each shipped config must build a network and produce action logits."""
    config = load_config(config_path)
    model = build_network(config)

    n_agents = config["num_agents"]
    fov = config["map_shape"]
    states = torch.randn(2, n_agents, 2, fov[0], fov[1])
    gso = torch.eye(n_agents).unsqueeze(0).repeat(2, 1, 1)

    with torch.no_grad():
        if config["net_type"] == "baseline":
            output = model(states)
        else:
            output = model(states, gso)

    assert output.shape == (2, n_agents, 5), (
        f"{config_path.name}: expected (2, {n_agents}, 5), got {tuple(output.shape)}"
    )
    assert torch.isfinite(output).all(), f"{config_path.name}: non-finite logits"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_env_settings_match_dataset(config_path):
    """The board and sensing settings must match the dataset directory name.

    Datasets are named {agents}_{obstacles}_{board}. A policy trained on one
    communication graph and evaluated on another scores far below its real
    performance, and nothing in the pipeline warns about it.
    """
    config = load_config(config_path)
    root_dir = config["train"]["root_dir"]
    name = Path(root_dir).name

    parts = name.split("_")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        pytest.skip(f"Dataset name {name!r} does not encode its settings")

    agents, obstacles, board = (int(p) for p in parts)
    assert config["num_agents"] == agents, f"{config_path.name}: agents"
    assert config["obstacles"] == obstacles, f"{config_path.name}: obstacles"
    assert config["board_size"] == [board, board], f"{config_path.name}: board_size"


def test_checkpoints_load_into_their_config():
    """Every shipped checkpoint must load into the architecture it declares."""
    trained_dir = Path(__file__).parent.parent / "trained_models"
    checkpoints = sorted(trained_dir.glob("*/model.pt"))
    if not checkpoints:
        pytest.skip("No trained models shipped")

    for checkpoint in checkpoints:
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            warnings.warn(f"{checkpoint.parent.name} ships no config.yaml")
            continue

        config = load_config(config_path)
        model = build_network(config)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
