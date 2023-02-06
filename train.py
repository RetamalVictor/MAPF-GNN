import sys

sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra")
sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra\models")

import os
import time
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from data_loader import GNNDataLoader


with open(r"configs\config_gnn_test.yaml", "r") as config_path:
    config = yaml.load(config_path, Loader=yaml.FullLoader)

net_type = config["net_type"]
exp_name = config["exp_name"]
tests_episodes = config["tests_episodes"]
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if net_type == "baseline":
    from models.framework_baseline import Network

elif net_type == "gnn":
    # from models.framework_gnn import Network
    from models.framework_gnn_message import Network


if not os.path.exists(rf"results\{exp_name}"):
    os.makedirs(rf"results\{exp_name}")

with open(rf"results\{exp_name}\config.yaml", "w") as config_path:
    yaml.dump(config, config_path)

if __name__ == "__main__":

    print("----- Training stats -----")
    data_loader = GNNDataLoader(config)
    model = Network(config)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.to(config["device"])

    losses = []
    success_rate_final = []
    flow_time_final = []

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

            total_loss = torch.zeros(1, requires_grad=True)
            for agent in range(trajectories.shape[1]):
                loss = criterion(output[:, agent, :], trajectories[:, agent].long())
                total_loss = total_loss + (loss / trajectories.shape[1])

            total_loss.backward()
            train_loss += total_loss
            optimizer.step()
        print(f"Loss: {train_loss.item()}")
        losses.append(train_loss.item())

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
        print(f"Success rate: {success_rate}")
        print(f"Flow time: {flow_time}")

    loss = np.array(losses)
    success_rate = np.array(success_rate_final)
    flow_time = np.array(flow_time_final)

    np.save(rf"results\{exp_name}\success_rate.npy", success_rate)
    np.save(rf"results\{exp_name}\flow_time.npy", flow_time)
    np.save(rf"results\{exp_name}\loss.npy", loss)

    torch.save(model.state_dict(), rf"results\{exp_name}\model.pt")
