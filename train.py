import argparse
import os
import time
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from data_loader import GNNDataLoader

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train MAPF-GNN model')
parser.add_argument('--config', type=str, default='configs/config_gnn.yaml',
                    help='Path to configuration file')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
args = parser.parse_args()

# Load configuration
config_path = Path(args.config)
with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

net_type = config["net_type"]
exp_name = config["exp_name"]
tests_episodes = config["tests_episodes"]
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if net_type == "baseline":
    from models.framework_baseline import Network

elif net_type == "gnn":
    # from models.framework_gnn import Network
    from models.framework_gnn_message import Network


results_dir = Path("results") / exp_name
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / "config.yaml", "w") as config_file:
    yaml.dump(config, config_file)

if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("----- Training stats -----")
    print(f"Using config: {args.config}")
    print(f"Random seed: {args.seed}")

    data_loader = GNNDataLoader(config)
    model = Network(config)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()

    model.to(config["device"])

    losses = []
    success_rate_final = []
    flow_time_final = []

    best_success_rate = 0.0

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}")

        # ##### Training #########
        model.train()
        train_loss = 0
        for i, (states, trajectories, gso) in enumerate(data_loader.train_loader):
            optimizer.zero_grad()
            states = states.to(config["device"])
            trajectories = trajectories.to(config["device"])
            gso = gso.to(config["device"])
            output = model(states, gso)

            # Efficient loss computation
            batch_size, n_agents, n_actions = output.shape
            output_flat = output.reshape(-1, n_actions)
            trajectories_flat = trajectories.long().reshape(-1)
            loss = criterion(output_flat, trajectories_flat)

            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            train_loss += loss.item()
            optimizer.step()
        avg_train_loss = train_loss / len(data_loader.train_loader)
        print(f"Loss: {avg_train_loss:.4f}")
        losses.append(avg_train_loss)

        ######### Validation #########
        val_loss = 0
        model.eval()
        success_rate = []
        flow_time = []
        for episode in range(tests_episodes):
            goals = create_goals(config["board_size"], config["num_agents"])
            obstacles = create_obstacles(config["board_size"], config["obstacles"])
            env = GraphEnv(config, goal=goals, obstacles=obstacles)
            emb = env.getEmbedding()
            obs = env.reset()
            for i in range(config["max_steps"]):
                fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
                gso = (
                    torch.tensor(obs["adj_matrix"])
                    .float()
                    .unsqueeze(0)
                    .to(config["device"])
                )
                with torch.no_grad():
                    action = model(fov, gso)
                action = action.cpu().squeeze(0).numpy()
                action = np.argmax(action, axis=1)
                obs, reward, done, info = env.step(action, emb)
                if done:
                    break

            metrics = env.computeMetrics()
            success_rate.append(metrics[0])
            flow_time.append(metrics[1])

        success_rate = np.mean(success_rate)
        flow_time = np.mean(flow_time)
        success_rate_final.append(success_rate)
        flow_time_final.append(flow_time)
        print(f"Success rate: {success_rate:.3f}")
        print(f"Flow time: {flow_time:.2f}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            torch.save(model.state_dict(), results_dir / "best_model.pt")
            print(f"New best model saved with success rate: {best_success_rate:.3f}")

        # Step the scheduler
        scheduler.step()

    loss = np.array(losses)
    success_rate = np.array(success_rate_final)
    flow_time = np.array(flow_time_final)

    np.save(results_dir / "success_rate.npy", success_rate)
    np.save(results_dir / "flow_time.npy", flow_time)
    np.save(results_dir / "loss.npy", loss)

    torch.save(model.state_dict(), results_dir / "model.pt")
