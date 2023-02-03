import sys
sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra")
sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra\models")

import os
import time
import yaml
import numpy as np
from tqdm import tqdm
from pprint import pprint
import torch
from torch import nn
from torch import optim

from grid.env_graph_gridv1 import GraphEnv, create_goals
from models.framework import Network
from data_loader import GNNDataLoader

from torch.utils.tensorboard import SummaryWriter
if not os.path.exists(r"results\2_0_8"):
    os.makedirs(r"results\2_0_8")

with open("config_1.yaml", 'r') as config_path:
    config = yaml.load(config_path, Loader=yaml.FullLoader)

exp_name = config["exp_name"]
writer = SummaryWriter(fr"results\2_0_8{exp_name}")

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    data_loader = GNNDataLoader(config)
    model = Network(config)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(config["device"])
    tests_episodes = config["tests_episodes"]
    
    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}")


        ######## Training #########
        model.train()
        train_loss = 0
        for i, (states, trajectories) in enumerate(data_loader.train_loader):
            optimizer.zero_grad()
            states = states.to(config["device"])
            trajectories = trajectories.to(config["device"])

            output = model(states)

            total_loss = torch.zeros(1,requires_grad=True)
            for agent in range(trajectories.shape[1]):
                loss = criterion(output[:,agent,:], trajectories[:,agent].long())
                total_loss = total_loss + (loss/trajectories.shape[1])

            total_loss.backward()
            train_loss += total_loss
            optimizer.step()
        print(f"Loss: {train_loss.item()}")
        writer.add_scalar('Loss/train', train_loss.item(), epoch)
        

        ######### Validation #########
        val_loss = 0
        model.eval()
        for episode in range(tests_episodes):
            goals = create_goals(config["board_size"], config["num_agents"])
            env = GraphEnv(config, goals)
            emb = env.getEmbedding()
            obs = env.reset()
            for i in range(config["max_steps"]):
                obs = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
                with torch.no_grad():
                    action = model(obs)
                action = action.cpu().squeeze(0).numpy()
                action = np.argmax(action, axis=1)
                obs, reward, done, info = env.step(action, emb)
                env.render()
                if done:
                    break
            
            metrics = env.computeMetrics()
            writer.add_scalar('Metrics/success_rate', metrics[0], episode)
            writer.add_scalar('Metrics/flow_time', metrics[1], episode)



############# New validation
        # print(f"Val loss: {val_loss/len(data_loader.valid_loader)}")
        # writer.add_scalar('Loss/val', val_loss/len(data_loader.valid_loader), epoch)
        