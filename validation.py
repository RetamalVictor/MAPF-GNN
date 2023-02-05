import sys

sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra")
sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra\models")
import yaml
import time
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

import torch
from torch import nn
from data_generation.record import make_env
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles
from models.framework_gnn import Network

with open("config_gnn_test.yaml", "r") as config_path:
    config = yaml.load(config_path, Loader=yaml.FullLoader)
config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_name = config["exp_name"]
tests_episodes = config["tests_episodes"]

model = Network(config)
model.to(config["device"])
model.load_state_dict(torch.load(rf"results\{exp_name}\model.pt"))
model.eval()

success_rate = np.zeros((tests_episodes, 1))
flow_time = np.zeros((tests_episodes, 1))

for episode in range(tests_episodes):
    obstacles = create_obstacles(config["board_size"], config["obstacles"])
    goals = create_goals(config["board_size"], config["num_agents"], obstacles)

    env = GraphEnv(config, goal=goals, obstacles=obstacles, sensing_range=4)
    emb = env.getEmbedding()
    obs = env.reset()
    for i in range(config["max_steps"] + 10):
        fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
        gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(config["device"])
        with torch.no_grad():
            action = model(fov, gso)
            action = action.cpu().squeeze(0).numpy()
        action = np.argmax(action, axis=1)

        obs, reward, done, info = env.step(action, emb)
        env.render(None)
        if done:
            print("All agents reached their goal\n")
            break
        if i == config["max_steps"] + 9:
            print("Max steps reached")

    metrics = env.computeMetrics()
    print(metrics)

    success_rate[episode] = metrics[0]
    flow_time[episode] = metrics[1]


print("Success rate: ", np.mean(success_rate))
print("Flow time: ", np.mean(flow_time))
