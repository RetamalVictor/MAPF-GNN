"""Export trained policies for the browser demo.

The demo at victor-retamal.com runs the policy client-side, so it needs the
weights in a layout a TypeScript or WGSL implementation can consume without
replicating PyTorch's conventions. This script is the entire coupling between
the two repositories: it writes a `.bin` per model plus a `config.json` manifest
and a `fixture.json` parity test into the site's asset directory.

Three things happen here that are worth knowing about.

**BatchNorm is folded into the preceding convolution.** The demo only ever runs
in eval mode, so carrying four extra tensors and a normalisation step would buy
nothing.

**Matrices are transposed at export, not at load.** PyTorch stores `nn.Linear`
weights as (out, in) and the graph filter as (in, K, out), and the model then
reshapes and permutes them at every forward pass. Doing that once here means the
browser code is a plain row-major `x @ W`, which is both faster and much harder
to get subtly wrong.

**Weights are fp32, not fp16.** Halving 350 KB is not worth the risk: folding
BatchNorm can inflate a convolution's dynamic range by whatever
`1/sqrt(running_var + eps)` happens to be, and the parity test wants headroom
well under its 1e-3 tolerance.

The message-passing architecture is deliberately not supported; see
`docs/findings.md` on its self-term.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from grid.env_vectorized import ACTION_DX, ACTION_DY, VectorizedGraphEnv

BN_EPS = 1e-5

# Conv index -> BatchNorm index inside the `convLayers` Sequential.
CONV_BN_PAIRS = ((0, 1), (3, 4), (6, 7))


def fold_batchnorm(weight, bias, bn_gamma, bn_beta, bn_mean, bn_var):
    """Absorb a BatchNorm into the convolution that feeds it.

    conv:  y = W x + b
    bn:    z = gamma (y - mean) / sqrt(var + eps) + beta

    so with s = gamma / sqrt(var + eps), the pair collapses to a convolution
    with weights `W * s` and bias `s (b - mean) + beta`.
    """
    scale = bn_gamma / torch.sqrt(bn_var + BN_EPS)
    return weight * scale.reshape(-1, 1, 1, 1), scale * (bias - bn_mean) + bn_beta


def build_tensors(state_dict, arch, filter_number):
    """Return the export layout: a list of (name, shape, float32 numpy array).

    Order is fixed and is what the browser reads back. Every matrix is stored so
    the consumer does `x @ W` with x as a row vector.
    """
    tensors = []

    def add(name, array):
        array = np.ascontiguousarray(array.detach().cpu().numpy(), dtype=np.float32)
        tensors.append((name, list(array.shape), array))

    for index, (conv, bn) in enumerate(CONV_BN_PAIRS, start=1):
        weight, bias = fold_batchnorm(
            state_dict[f"convLayers.{conv}.weight"],
            state_dict[f"convLayers.{conv}.bias"],
            state_dict[f"convLayers.{bn}.weight"],
            state_dict[f"convLayers.{bn}.bias"],
            state_dict[f"convLayers.{bn}.running_mean"],
            state_dict[f"convLayers.{bn}.running_var"],
        )
        add(f"conv{index}.w", weight)  # (out, in, 3, 3)
        add(f"conv{index}.b", bias)

    # nn.Linear stores (out, in); the browser wants (in, out).
    add("enc.w", state_dict["compressMLP.0.weight"].t())
    add("enc.b", state_dict["compressMLP.0.bias"])

    if arch == "gcn":
        # GCNLayer.forward reshapes W from (in, K, out) to (out, K*in) and then
        # transposes it. That is a reinterpretation of the stored buffer rather
        # than a mathematical transpose, so reproduce it exactly rather than
        # reasoning about what it "should" be.
        gnn_w = state_dict["GFL.0.W"]
        in_features, k, out_features = gnn_w.shape
        if k != filter_number:
            raise ValueError(f"config says K={filter_number}, checkpoint has K={k}")
        add("gnn.w", gnn_w.reshape(out_features, k * in_features).t())

    add("act.w", state_dict["actionMLP.0.weight"].t())
    add("act.b", state_dict["actionMLP.0.bias"])
    return tensors


def write_bin(path, tensors):
    """Concatenate the tensors as little-endian float32 and return the manifest."""
    manifest, offset, blobs = [], 0, []
    for name, shape, array in tensors:
        flat = array.reshape(-1)
        manifest.append(
            {"name": name, "shape": shape, "offset": offset, "count": int(flat.size)}
        )
        blobs.append(flat)
        offset += int(flat.size)

    payload = np.concatenate(blobs).astype("<f4", copy=False)
    path.write_bytes(payload.tobytes())
    return manifest, offset


def load_network(config, arch):
    """Instantiate the architecture the config names."""
    if arch == "gcn":
        from models.framework_gnn import Network
    elif arch == "baseline":
        from models.framework_baseline import Network
    else:
        raise ValueError(f"unsupported architecture: {arch}")
    return Network(config)


def trained_obstacle_range(config):
    """The obstacle densities a model was actually trained on.

    Read from the dataset directory names, which encode {agents}_{obstacles}_
    {board}. Shipping this with the weights means the demo can say honestly
    whether a slider position is inside the training distribution, instead of
    carrying a hardcoded guess that goes stale the moment the curriculum
    changes - which is exactly what happened.
    """
    roots = config.get("train", {}).get("root_dir", [])
    if isinstance(roots, str):
        roots = [roots]

    fractions = []
    for root in roots:
        parts = Path(root).name.split("_")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            _, obstacles, board = (int(p) for p in parts)
            fractions.append(obstacles / (board * board))
    return [min(fractions), max(fractions)] if fractions else None


def detect_arch(config, name):
    net_type = config.get("net_type", "baseline")
    if net_type == "baseline":
        return "baseline"
    if config.get("msg_type") == "message":
        raise SystemExit(
            f"{name} uses the message-passing layer, whose self-term reads a "
            "scrambled global mixture of every agent's features rather than the "
            "agent's own (docs/findings.md, 'The self-term'). It is not "
            "decentralised, so it must not back a demo about decentralised "
            "control. Export the GCN checkpoint instead."
        )
    return "gcn"


def sample_instance(rng, board, num_agents, num_obstacles):
    """Draw obstacles, starts and goals from distinct free cells."""
    cells = rng.choice(board * board, size=num_obstacles + 2 * num_agents, replace=False)
    coords = np.stack([cells % board, cells // board], axis=1)  # x, y
    obstacles = coords[:num_obstacles]
    starts = coords[num_obstacles : num_obstacles + num_agents]
    goals = coords[num_obstacles + num_agents :]
    return obstacles, starts, goals


def make_fixture(networks, board, num_agents, num_obstacles, steps, sensing_range,
                 max_neighbours, seed):
    """Roll the policies out deterministically and record every intermediate.

    The browser reproduces this from the same starts, goals and obstacles. Field
    of view and adjacency are recorded per step so a failure localises to the
    environment or the forward pass instead of just saying "the demo is wrong".
    """
    rng = np.random.default_rng(seed)
    obstacles, starts, goals = sample_instance(rng, board, num_agents, num_obstacles)

    # max_neighbours must be carried explicitly: VectorizedGraphEnv defaults it
    # to 4, so leaving it out silently generates a capped fixture for a manifest
    # that declares an uncapped graph. The parity test catches it, loudly.
    config = {
        "board_size": [board, board],
        "num_agents": num_agents,
        "max_time": steps + 1,
        "sensing_range": sensing_range,
        "max_neighbours": max_neighbours or None,
    }

    fixture = {
        "note": (
            "Ground truth from the PyTorch model on CPU. Rebuild the same "
            "instance in the browser, take the argmax at every step, and expect "
            "identical positions and actions."
        ),
        "board": board,
        "num_agents": num_agents,
        "sensing_range": sensing_range,
        "obstacles": obstacles.tolist(),
        "starts": starts.tolist(),
        "goals": goals.tolist(),
        "models": {},
    }

    for key, (net, arch) in networks.items():
        env = VectorizedGraphEnv(
            config,
            goals=goals[None],
            obstacles=obstacles[None],
            starting_positions=starts[None],
            device="cpu",
        )
        observation = env.get_observations()
        record = []
        for _ in range(steps):
            with torch.no_grad():
                # The baseline has no graph to consume, so it takes one argument.
                if arch == "baseline":
                    logits = net(observation["fov"])
                else:
                    logits = net(observation["fov"], observation["adj_matrix"])
            actions = logits.argmax(dim=2)
            record.append(
                {
                    "positions": env.get_positions()[0].tolist(),
                    "fov": np.round(observation["fov"][0].numpy(), 4).tolist(),
                    "adjacency": observation["adj_matrix"][0].numpy().astype(int).tolist(),
                    "logits": np.round(logits[0].numpy(), 6).tolist(),
                    "actions": actions[0].tolist(),
                }
            )
            observation, _ = env.step(actions)
        record.append({"positions": env.get_positions()[0].tolist()})
        fixture["models"][key] = {"steps": record}

    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("../retamalvictor.github.io/public/assets/models/mapf"),
        help="asset directory in the site repository",
    )
    # The demo shows the communicating policy only. The baseline is still
    # exported because the parity fixture uses the disagreement between the two
    # to prove it is crowded enough to detect a missing communication path - a
    # check worth far more than the 121 KB it costs.
    parser.add_argument("--comm-config", default="configs/config_gnn_k3_dense_r8.yaml")
    parser.add_argument("--comm-checkpoint", default="results/gnn_k3_dense_r8/best_model.pt")
    parser.add_argument("--nocomm-config", default="configs/config_baseline_dense.yaml")
    parser.add_argument(
        "--nocomm-checkpoint", default="results/baseline_dense/best_model.pt"
    )
    # A deliberately crowded instance. On a sparse board the two policies choose
    # identical actions, so the fixture would pass for an implementation that
    # ignored the communication graph entirely. At this density they disagree on
    # 12 of 96 actions, which is what makes the parity test worth running.
    parser.add_argument("--fixture-agents", type=int, default=12)
    parser.add_argument("--fixture-board", type=int, default=12)
    parser.add_argument("--fixture-obstacles", type=int, default=6)
    parser.add_argument("--fixture-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    sources = {
        "comm": (args.comm_config, args.comm_checkpoint),
        "nocomm": (args.nocomm_config, args.nocomm_checkpoint),
    }

    manifest_models, networks, shared = {}, {}, None
    trained_obstacles = None
    for key, (config_path, checkpoint_path) in sources.items():
        config = yaml.safe_load(Path(config_path).read_text())
        arch = detect_arch(config, key)

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        net = load_network(config, arch)
        net.load_state_dict(state_dict)
        net.eval()
        networks[key] = (net, arch)

        filter_number = config.get("graph_filters", [0])[0]
        tensors = build_tensors(state_dict, arch, filter_number)
        binary = args.out / f"{key}.bin"
        entries, total = write_bin(binary, tensors)

        manifest_models[key] = {
            "file": binary.name,
            "arch": arch,
            "source": str(checkpoint_path).replace("\\", "/"),
            "trained_agents": config["num_agents"],
            "trained_board": config["board_size"][0],
            "channels": [2] + config["channels"],
            "encoder_dim": config["encoder_dims"][-1],
            "node_dim": config["node_dims"][-1] if arch == "gcn" else None,
            "filter_number": filter_number if arch == "gcn" else None,
            "parameters": total,
            "tensors": entries,
        }
        print(f"{key:7s} {arch:9s} {total:>7,} floats  {binary.stat().st_size/1024:6.1f} KB")

        # Both panels run in the same world - only the policy differs - so the
        # graph settings have to match. A baseline ignores the graph, but the
        # environment still builds one for the communicating model beside it.
        trained_obstacles = trained_obstacle_range(config) or trained_obstacles
        candidate = (
            config["sensing_range"],
            tuple(config["map_shape"]),
            config.get("max_neighbours", 4) or 0,
        )
        if shared is None:
            shared = candidate
        elif shared != candidate:
            raise SystemExit(
                f"models disagree about the world: {shared} vs {candidate} "
                "(sensing range, FOV shape, neighbour cap)"
            )

    sensing_range, map_shape, max_neighbours = shared
    manifest = {
        "note": (
            "Weights for the MAPF-GNN browser demo. float32, little-endian, "
            "row-major; every matrix is stored so the consumer computes x @ W. "
            "BatchNorm is folded into the preceding convolution."
        ),
        "dtype": "float32",
        "fov": map_shape,
        "pad": 3,
        "sensing_range": sensing_range,
        # 0 means uncapped: aggregate over every robot in range. Capping it also
        # makes a wider sensing range pointless, since it swaps near neighbours
        # for far ones rather than adding any.
        "max_neighbours": max_neighbours,
        "actions": {
            "names": ["idle", "right", "up", "left", "down"],
            "dx": list(ACTION_DX),
            "dy": list(ACTION_DY),
        },
        "defaults": {
            # Both must match the article: temperature has a real optimum, and
            # the horizon has to scale with the board or it becomes a
            # brute-force budget that flatters every policy.
            "temperature": 3.0,
            "horizon_factor": 3,
            # 100, not 40. At 40 robots the communicating and non-communicating
            # policies score 0.93 against 0.94 - a reader watching the default
            # sees two identical panels and concludes the demo is broken. The
            # gap only opens at 100: 0.60 against 0.21.
            "agents": 100,
            "board": 63,
            "obstacle_fraction": 0.02,
            # What the slider can claim is "as trained".
            "trained_obstacles": trained_obstacles,
        },
        "models": manifest_models,
    }
    (args.out / "config.json").write_text(json.dumps(manifest, indent=2))

    fixture = make_fixture(
        networks,
        board=args.fixture_board,
        num_agents=args.fixture_agents,
        num_obstacles=args.fixture_obstacles,
        steps=args.fixture_steps,
        sensing_range=sensing_range,
        max_neighbours=max_neighbours,
        seed=args.seed,
    )
    (args.out / "fixture.json").write_text(json.dumps(fixture))
    print(f"fixture {args.fixture_agents} agents, {args.fixture_steps} steps -> {args.out}")


if __name__ == "__main__":
    main()
