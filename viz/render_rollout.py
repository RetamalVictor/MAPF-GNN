"""Render policy rollouts as GIFs.

Colour encodes agent *state*, not identity: twenty arbitrary hues would be
decoration, whereas moving / arrived / stalled is the thing the story is about.
Communication edges are drawn from the adjacency matrix the policy actually
consumed, so what you see is the graph the GNN saw.

Usage:
    python viz/render_rollout.py --model results/gnn_msg_k3 --agents 20 --out viz/out
    python viz/render_rollout.py --compare results/baseline_a10b20 results/gnn_msg_k3_a10b20 \
        --agents 20 --out viz/out
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def _font(size, bold=False):
    """A real UI face; PIL's built-in bitmap font is unreadably small."""
    for name in (("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()

sys.path.append(str(Path(__file__).parent.parent))

from benchmarks.benchmark_transfer import build_network, make_episodes
from grid.env_vectorized import VectorizedGraphEnv

# Light surface, from the reference palette
SURFACE = (252, 252, 251)
GRID = (225, 224, 217)
OBSTACLE = (110, 108, 102)
LINK = (137, 135, 129)
INK = (11, 11, 11)
MUTED = (137, 135, 129)

MOVING = (42, 120, 214)    # series 1 blue
ARRIVED = (12, 163, 12)    # status good
STALLED = (208, 59, 59)    # status critical

CELL = 18
MARGIN = 26
STALL_AFTER = 6  # steps without moving before an agent counts as stalled

# Readable names for the figures
LABELS = {
    "baseline_a10b20": "No communication (CNN + MLP)",
    "gnn_msg_k3_a10b20": "With communication (GNN, K=3)",
    "gnn_k3_a10b20": "With communication (GCN, K=3)",
    "baseline": "No communication (CNN + MLP)",
    "gnn_msg_k3": "With communication (GNN, K=3)",
}


def rollout_frames(model, config, board, agents, obstacles, seed, max_steps, device):
    """Run one episode, returning per-step positions, goals, graph and status."""
    obs_arr, goals, starts = make_episodes(board, agents, obstacles, 1, seed)
    env = VectorizedGraphEnv(
        dict(config, board_size=[board, board]),
        goals=goals, obstacles=obs_arr, starting_positions=starts, device=device,
    )
    obs = env.get_observations()
    is_gnn = config["net_type"] != "baseline"

    history = []
    still_for = np.zeros(agents, dtype=int)
    previous = env.get_positions()[0].cpu().numpy()

    for _ in range(max_steps):
        positions = env.get_positions()[0].cpu().numpy()
        at_goal = env.at_goal()[0].cpu().numpy()
        still_for = np.where((positions == previous).all(axis=1), still_for + 1, 0)
        history.append({
            "positions": positions.copy(),
            "adjacency": obs["adj_matrix"][0].cpu().numpy().copy(),
            "at_goal": at_goal.copy(),
            "stalled": (still_for >= STALL_AFTER) & ~at_goal,
        })
        if at_goal.all():
            break

        previous = positions
        with torch.no_grad():
            logits = model(obs["fov"], obs["adj_matrix"]) if is_gnn else model(obs["fov"])
        obs, _ = env.step(logits.argmax(dim=2))

    return history, goals[0], obs_arr[0]


def draw_frame(frame, goals, obstacles, board, title=None, subtitle=None, legend=False):
    width = board * CELL + 2 * MARGIN
    header = 46 if title else 0
    footer = 30 if legend else 0
    img = Image.new("RGB", (width, board * CELL + 2 * MARGIN + header + footer), SURFACE)
    d = ImageDraw.Draw(img)
    top = MARGIN + header
    title_font, label_font = _font(17, bold=True), _font(14)

    def cell(x, y):
        return MARGIN + x * CELL, top + y * CELL

    for i in range(board + 1):
        d.line([(MARGIN + i * CELL, top), (MARGIN + i * CELL, top + board * CELL)], fill=GRID)
        d.line([(MARGIN, top + i * CELL), (MARGIN + board * CELL, top + i * CELL)], fill=GRID)

    for ox, oy in obstacles:
        x0, y0 = cell(ox, oy)
        d.rectangle([x0 + 1, y0 + 1, x0 + CELL - 1, y0 + CELL - 1], fill=OBSTACLE)

    # Goals sit under the agents as hollow rings
    for gx, gy in goals:
        x0, y0 = cell(gx, gy)
        d.rectangle([x0 + 4, y0 + 4, x0 + CELL - 4, y0 + CELL - 4], outline=MUTED, width=2)

    # Communication edges, drawn before the agents so nodes sit on top
    positions = frame["positions"]
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if frame["adjacency"][i, j]:
                xi, yi = cell(*positions[i])
                xj, yj = cell(*positions[j])
                d.line([(xi + CELL // 2, yi + CELL // 2), (xj + CELL // 2, yj + CELL // 2)],
                       fill=LINK, width=1)

    for i, (px, py) in enumerate(positions):
        x0, y0 = cell(px, py)
        if frame["at_goal"][i]:
            colour = ARRIVED
        elif frame["stalled"][i]:
            colour = STALLED
        else:
            colour = MOVING
        d.ellipse([x0 + 3, y0 + 3, x0 + CELL - 3, y0 + CELL - 3], fill=colour)

    if title:
        d.text((MARGIN, 14), title, fill=INK, font=title_font)
        if subtitle:
            right = d.textlength(subtitle, font=label_font)
            d.text((width - MARGIN - right, 17), subtitle, fill=MUTED, font=label_font)

    if legend:
        y = top + board * CELL + 12
        x = MARGIN
        for colour, label in ((MOVING, "moving"), (ARRIVED, "at goal"),
                              (STALLED, f"stalled >{STALL_AFTER} steps")):
            d.ellipse([x, y + 3, x + 11, y + 14], fill=colour)
            d.text((x + 17, y + 1), label, fill=MUTED, font=label_font)
            x += int(d.textlength(label, font=label_font)) + 44
        d.line([(x, y + 8), (x + 22, y + 8)], fill=LINK, width=2)
        d.text((x + 28, y + 1), "can communicate", fill=MUTED, font=label_font)
    return img


def save_gif(frames, path, ms=140, hold=12):
    frames = frames + [frames[-1]] * hold  # linger on the final state
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=ms, loop=0, optimize=True)
    print(f"wrote {path}  ({len(frames)} frames, {path.stat().st_size / 1024:.0f} KB)")


def load(model_dir, device):
    from evaluate import _load_yaml

    model_dir = Path(model_dir)
    with open(model_dir / "config.yaml") as f:
        config = _load_yaml(f)
    config["device"] = device
    model = build_network(config).to(device).eval()
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))
    return model, config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"))
    p.add_argument("--agents", type=int, default=20)
    p.add_argument("--board", type=int, default=28)
    p.add_argument("--obstacles", type=int, default=8)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=60)
    p.add_argument("--out", default="viz/out")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.compare:
        panels = []
        for model_dir in args.compare:
            model, config = load(model_dir, device)
            history, goals, obstacles = rollout_frames(
                model, config, args.board, args.agents, args.obstacles,
                args.seed, args.max_steps, device)
            panels.append((Path(model_dir).name, history, goals, obstacles))

        length = max(len(h) for _, h, _, _ in panels)
        frames = []
        for t in range(length):
            images = []
            for name, history, goals, obstacles in panels:
                frame = history[min(t, len(history) - 1)]
                done = int(frame["at_goal"].sum())
                images.append(draw_frame(
                    frame, goals, obstacles, args.board, legend=True,
                    title=LABELS.get(name, name),
                    subtitle=f"{done}/{args.agents} at goal   step {min(t, len(history)-1)}"))
            combined = Image.new("RGB", (sum(i.width for i in images), images[0].height), SURFACE)
            x = 0
            for image in images:
                combined.paste(image, (x, 0))
                x += image.width
            frames.append(combined)
        save_gif(frames, out / f"compare_{args.agents}agents.gif")
        return

    model, config = load(args.model, device)
    history, goals, obstacles = rollout_frames(
        model, config, args.board, args.agents, args.obstacles,
        args.seed, args.max_steps, device)
    name = Path(args.model).name
    frames = [
        draw_frame(f, goals, obstacles, args.board, title=LABELS.get(name, name), legend=True,
                   subtitle=f"{int(f['at_goal'].sum())}/{args.agents} at goal   step {t}")
        for t, f in enumerate(history)
    ]
    save_gif(frames, out / f"{name}_{args.agents}agents.gif")


if __name__ == "__main__":
    main()
