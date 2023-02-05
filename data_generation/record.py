import sys

sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra")

import os
import yaml
import numpy as np
from grid.env_graph_gridv1 import GraphEnv
import matplotlib.pyplot as plt


def make_env(pwd_path, config):
    with open(os.path.join(pwd_path, "input.yaml")) as input_params:
        params = yaml.load(input_params, Loader=yaml.FullLoader)
    nb_agents = len(params["agents"])
    dimensions = params["map"]["dimensions"]
    obstacles = params["map"]["obstacles"]
    starting_pos = np.zeros((nb_agents, 2), dtype=np.int32)
    goals = np.zeros((nb_agents, 2), dtype=np.int32)
    obstacles_list = np.zeros((len(obstacles), 2), dtype=np.int32)
    for i in range(len(obstacles)):
        obstacles_list[i, :] = np.array([int(obstacles[i][0]), int(obstacles[i][1])])

    for d, i in zip(params["agents"], range(0, nb_agents)):
        #   name = d["name"]
        starting_pos[i, :] = np.array([int(d["start"][0]), int(d["start"][1])])
        goals[i, :] = np.array([int(d["goal"][0]), int(d["goal"][1])])

    env = GraphEnv(
        config=config,
        goal=goals,
        board_size=int(dimensions[0]),
        starting_positions=starting_pos,
        obstacles=obstacles_list,
        sensing_range=config["sensor_range"],
    )
    return env


def record_env(path, config):
    cases = os.listdir(path)
    t = np.zeros(len(cases))

    for i in range(len(cases) - 1):
        trayectory = np.load(
            os.path.join(path, rf"case_{i}\trajectory.npy"), allow_pickle=True
        )
        t[i] = trayectory.shape[1]

    print(f"max steps {np.max(t)}")
    print(f"min steps {np.min(t)}")
    print(f"mean steps {np.mean(t)}")
    with open(os.path.join(path, "stats.txt"), "w") as f:
        f.write(f"max steps {np.max(t)}\n")
        f.write(f"min steps {np.min(t)}\n")
        f.write(f"mean steps {np.mean(t)}\n")

    # mx = int(np.max(t))
    # print(f"Max step: {mx}")
    print("Recording states...")
    for timestep in range(len(cases) - 1):
        agent_nb = trayectory.shape[0]
        env = make_env(os.path.join(path, rf"case_{timestep}"), config)
        # mx = env.min_time
        trayectory = np.load(
            os.path.join(path, rf"case_{timestep}\trajectory.npy"), allow_pickle=True
        )
        trayectory = trayectory[:, 1:]
        recordings = np.zeros(
            (trayectory.shape[1], agent_nb, 2, 5, 5)
        )  # timestep, agents, channels of FOV, dimFOVx, dimFOVy
        adj_record = np.zeros((trayectory.shape[1], agent_nb, agent_nb, agent_nb))
        assert (
            agent_nb == env.nb_agents
        ), rf"Trayectory has {agent_nb} agents, env expects {env.nb_agents}"
        # if trayectory.shape[1] < mx:
        #     continue
        #     trayectory = np.pad(trayectory,[(0,0), (0, mx - trayectory.shape[1])], mode='constant')
        obs = env.reset()
        emb = np.ones(env.nb_agents)
        for i in range(trayectory.shape[1]):
            recordings[i, :, :, :, :] = obs["fov"]
            adj_record[i, :, :, :] = obs["adj_matrix"]

            actions = trayectory[:, i]
            obs, _, _, _ = env.step(actions, emb)

        recordings[i, :, :, :, :] = obs["fov"]
        adj_record[i, :, :, :] = obs["adj_matrix"]

        np.save(os.path.join(path, rf"case_{timestep}\states.npy"), recordings)
        np.save(os.path.join(path, rf"case_{timestep}\gso.npy"), adj_record)
        np.save(
            os.path.join(path, rf"case_{timestep}\trajectory_record.npy"), trayectory
        )
        if timestep % 25 == 0:
            print(f"Recorded -- [{timestep}/{len(cases)}]")
    print(f"Recorded -- [{timestep}/{len(cases)}] --- completed")


if __name__ == "__main__":

    # total=200
    pwd_path = rf"dataset\5_7_16\test"
    record_env(pwd_path)
