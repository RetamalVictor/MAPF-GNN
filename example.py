#!/usr/bin/env python3
"""
Simple demo script for MAPF-GNN
Runs a single episode with visualization to demonstrate the model.
"""

import yaml
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles


def main():
    """Run a simple demonstration episode."""
    # Load configuration
    config_path = "configs/config_gnn.yaml"
    print(f"Loading configuration from {config_path}...")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    print(f"Using device: {device}\n")

    # Load model
    exp_name = config["exp_name"]
    net_type = config["net_type"]
    msg_type = config.get("msg_type", "gcn")

    print(f"Loading {net_type} model ({msg_type})...")

    if net_type == "gnn":
        if msg_type == "message":
            from models.framework_gnn_message import Network
        else:
            from models.framework_gnn import Network
    elif net_type == "baseline":
        from models.framework_baseline import Network

    model = Network(config)
    model.to(device)

    # Load weights
    model_path = Path("trained_models") / exp_name / "model.pt"
    if not model_path.exists():
        model_path = Path("results") / exp_name / "model.pt"

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}, using random weights")

    model.eval()

    # Create environment
    print(f"\nSetting up environment:")
    print(f"  Board size: {config['board_size']}")
    print(f"  Agents: {config['num_agents']}")
    print(f"  Obstacles: {config['obstacles']}")

    obstacles = create_obstacles(config["board_size"], config["obstacles"])
    goals = create_goals(config["board_size"], config["num_agents"], obstacles)

    env = GraphEnv(
        config,
        goal=goals,
        obstacles=obstacles,
        sensing_range=config.get("sensing_range", 6)
    )

    # Enable interactive plotting
    plt.ion()

    # Run episode
    print(f"\nRunning demonstration episode...")
    print("Close the visualization window to end.\n")

    emb = env.getEmbedding()
    obs = env.reset()

    max_steps = config["max_steps"] + 10

    for step in range(max_steps):
        # Prepare inputs
        fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(device)
        gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(device)

        # Get action from model
        with torch.no_grad():
            if net_type == "gnn":
                action = model(fov, gso)
            else:
                action = model(fov)
            action = action.cpu().squeeze(0).numpy()

        action = np.argmax(action, axis=1)

        # Step environment
        obs, reward, done, info = env.step(action, emb)

        # Render
        env.render(None)

        # Check if done
        if done:
            print(f"✓ Success! All agents reached their goals in {step+1} steps")
            break

        if step == max_steps - 1:
            print(f"⚠ Max steps ({max_steps}) reached")

    # Compute and display metrics
    metrics = env.computeMetrics()
    success_rate = metrics[0]
    flow_time = metrics[1]

    print(f"\nEpisode Results:")
    print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Flow Time: {flow_time}")
    print(f"  Steps Taken: {step+1}")

    # Keep plot open
    print("\nVisualization window will stay open. Press Ctrl+C or close window to exit.")
    try:
        plt.ioff()
        plt.show()
    except KeyboardInterrupt:
        print("\nDemo ended by user")


if __name__ == "__main__":
    main()
