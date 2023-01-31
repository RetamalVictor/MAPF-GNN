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

from grid.env_graph_gridv1 import GraphEnv
from models.framework import Network
from data_loader import GNNDataLoader

from torch.utils.tensorboard import SummaryWriter
if not os.path.exists(r"results\2_0_8"):
    os.makedirs(r"results\2_0_8")

with open("config_1.yaml", 'r') as config_path:
    config = yaml.load(config_path, Loader=yaml.FullLoader)
writer = SummaryWriter(r"results\2_0_8")

config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    data_loader = GNNDataLoader(config)
    model = Network(config)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    model.to(config["device"])
    
    for epoch in range(100):
        print(f"Epoch {epoch}")

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
        # print(f"Loss: {train_loss/len(data_loader.train_loader)}")
        # writer.add_scalar('Loss/train', train_loss/len(data_loader.train_loader), epoch)
        
        val_loss = 0
        model.eval()
        for i, (states, trajectories) in enumerate(data_loader.valid_loader):
            states = states.to(config["device"])
            trajectories = trajectories.float().to(config["device"])
            with torch.no_grad():
              output = model(states)
              total_loss = 0
              for agent in range(trajectories.shape[1]):
                  loss = criterion(output[:,agent,:].float(), trajectories[:,agent].long())
                  total_loss += (loss.item()/trajectories.shape[1])
              val_loss += total_loss
        print(f"Val loss: {val_loss}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        # print(f"Val loss: {val_loss/len(data_loader.valid_loader)}")
        # writer.add_scalar('Loss/val', val_loss/len(data_loader.valid_loader), epoch)
        