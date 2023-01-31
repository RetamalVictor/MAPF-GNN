import sys
sys.path.append(r"C:\Users\victo\Desktop\VU master\MLGP\Extra")

import os
import yaml
import numpy as np
from grid.env_graph_gridv1 import GraphEnv


def make_env(pwd_path):
    with open(os.path.join(pwd_path,"input.yaml")) as input_params:
            params = yaml.load(input_params, Loader=yaml.FullLoader)
    nb_agents = len(params["agents"])
    dimensions  = params["map"]["dimensions"]
    obstacles   = params["map"]["obstacles"]
    starting_pos = np.zeros((nb_agents,2), dtype=np.int32)
    goals = np.zeros((nb_agents,2), dtype=np.int32)
    

    for d, i in zip(params["agents"], range(0, nb_agents)):
    #   name = d["name"]
      starting_pos[i,:] = np.array([int(d["start"][0]), int(d["start"][1])])
      goals[i,:]        = np.array([int(d["goal"][0]), int(d["goal"][1])])
    
    env = GraphEnv(
        nb_agents=nb_agents,
        goal=goals,
        board_size=int(dimensions[0]),
        starting_positions=starting_pos
        )
    return env


def record_env(path):
    cases = os.listdir(path)
    t = np.zeros(len(cases))
    
    for i in range(len(cases)):
        trayectory = np.load(os.path.join(path,fr"case_{i}\trajectory.npy"), allow_pickle=True)
        t[i] = trayectory.shape[1]
    # print(f"max steps {np.max(t)}")
    mx = int(np.max(t))
    print(f"Max step: {mx}")
    print("Recording states...")
    for timestep in range(len(cases)):
            agent_nb = trayectory.shape[0]
            recordings = np.zeros((mx+1, agent_nb, 2, 5, 5)) # timestep, agents, channels of FOV, dimFOVx, dimFOVy
            emb_record = np.zeros((mx+1, agent_nb,1))
            adj_record = np.zeros((mx+1, agent_nb, agent_nb,agent_nb))
            env = make_env(os.path.join(path, fr"case_{timestep}"))
            trayectory = np.load(os.path.join(path,fr"case_{timestep}\trajectory.npy"), allow_pickle=True)
            assert agent_nb == env.nb_agents, fr"Trayectory has {agent_nb} agents, env expects {env.nb_agents}"
            if trayectory.shape[1] != mx:
                trayectory = np.pad(trayectory,[(0,0), (0, mx - trayectory.shape[1])], mode='constant')
            obs = env.reset()
            emb = np.ones(env.nb_agents)
            for i in range(mx):
                # emb_record[i, :, :] = obs["embeddings"].reshape(2,1)
                recordings[i,:,:,:,:] = obs["fov"]
                adj_record[i, :, :, :] = obs["adj_matrix"]
                
                actions = trayectory[:,i]
                obs, _, _, _ = env.step(actions, emb)

            # emb_record[i, :, :] = obs["embeddings"].reshape(2,1)
            recordings[i,:,:,:,:] = obs["fov"]
            adj_record[i, :, :, :] = obs["adj_matrix"]

            np.save(os.path.join(path,fr"case_{timestep}\states.npy"), recordings)
            # np.save(os.path.join(path,fr"case_{timestep}\embeddings.npy"), emb_record)
            np.save(os.path.join(path,fr"case_{timestep}\gso.npy"), adj_record)
            np.save(os.path.join(path,fr"case_{timestep}\trajectory.npy"), trayectory)
            if timestep%25 == 0:
                print(f"Recorded -- [{timestep}/{len(cases)}]")
    print(f"Recorded -- [{timestep}/{len(cases)}] --- completed")





if __name__ == "__main__":
    
    # total=200
    pwd_path = fr"dataset\2_0_8\train"
    record_env(pwd_path)
