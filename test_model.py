import sys
sys.path.append(r'..\Extra')

import time
import numpy as np
from pprint import pprint
from grid.env_graph_grid import GraphEnv

from models.gnn import GCNLayer
from models.gnn_torch import GraphFilter

import torch
import torch.nn as nn


int_features = 1
out_features = 1
filte_taps = 20
edge_features= 1
bias= True

print("Init [OK]")

if __name__ == "__main__":
    agents = 20
    b_size = 100
    sensing = (b_size / 3)

    env = GraphEnv(agents, board_size=b_size, sensing_range=sensing)
    emb = np.ones(agents).reshape((agents,1))
    emb_torch = torch.from_numpy(emb).unsqueeze(0).double()
    gcn = GraphFilter(G=emb_torch.shape[1], F=emb_torch.shape[1], K=3, E=1, bias=False)
    print(f"GNN: {gcn}")
    obs = env.reset()
    test = [200]
    
    for le in range(len(test)):
        obs = env.reset()
        start = time.time()
        for i in range(test[le]):
            actions = np.random.randint(0,4,size=(agents))

            obs, _, _, _ = env.step(actions, emb_torch.detach().numpy().squeeze(0))
            gso = torch.from_numpy(obs["adj_matrix"]).unsqueeze(0)
            gcn.addGSO(gso)
            emb_torch = torch.from_numpy(emb).unsqueeze(0).double()
            emb_torch = gcn.forward(emb_torch)
            env.render(0,True)

        print(f"Simulation {test[le]} time: {time.time() - start}")
        print(f"Final embedings\n{emb_torch.detach().numpy().squeeze(0)}")
            