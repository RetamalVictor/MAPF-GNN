import yaml
import argparse
import numpy as np
from pathlib import Path

import torch
from torch import nn
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_gnn.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as config_path:
        config = yaml.load(config_path, Loader=yaml.FullLoader)

    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = config["exp_name"]
    tests_episodes = config["tests_episodes"]
    net_type = config["net_type"]
    msg_type = config["msg_type"]
    if net_type == "gnn":
        if msg_type == "message":
            from models.framework_gnn_message import Network
        else:
            from models.framework_gnn import Network

    if net_type == "baseline":
        from models.framework_baseline import Network

    success_rate = np.zeros((tests_episodes, 1))
    flow_time = np.zeros((tests_episodes, 1))
    all_goals = 0

    model = Network(config)
    model.to(config["device"])

    # Load model from trained_models directory or results directory
    model_path = Path("trained_models") / exp_name / "model.pt"
    if not model_path.exists():
        model_path = Path("results") / exp_name / "model.pt"

    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=config["device"]))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}")

    model.eval()
    for episode in range(tests_episodes):
        obstacles = create_obstacles(config["board_size"], config["obstacles"])
        goals = create_goals(config["board_size"], config["num_agents"], obstacles)

        env = GraphEnv(config, goal=goals, obstacles=obstacles, sensing_range=4)
        emb = env.getEmbedding()
        obs = env.reset()
        for i in range(config["max_steps"] + 10):
            fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(config["device"])
            gso = (
                torch.tensor(obs["adj_matrix"])
                .float()
                .unsqueeze(0)
                .to(config["device"])
            )
            with torch.no_grad():
                if net_type == "gnn":
                    action = model(fov, gso)
                if net_type == "baseline":
                    action = model(fov)  # , gso)
                action = action.cpu().squeeze(0).numpy()
            action = np.argmax(action, axis=1)

            obs, reward, done, info = env.step(action, emb)
            env.render(None)  # change to agentId = 0 to see the agent 0 communication
            if done:
                print("All agents reached their goal\n")
                all_goals += 1
                break
            if i == config["max_steps"] + 9:
                print("Max steps reached")

        metrics = env.computeMetrics()

        success_rate[episode] = metrics[0]
        flow_time[episode] = metrics[1]

    print("All goals: ", all_goals)
    print("All goals  mean: ", all_goals / tests_episodes)
    print("Success rate mean: ", np.mean(success_rate))
    print("Success rate std: ", np.std(success_rate))
    print("Flow time max: ", np.max(flow_time))
    print("Flow time: ", np.mean(flow_time))
    print("Flow time std: ", np.std(flow_time))
