import sys
sys.path.append(r'..\Extra')

from pprint import pprint
import numpy as np
from models.gnn import GCNLayer
from env_graph_gridv1 import GraphEnv
import time

if __name__ == "__main__":
    agents = 16
    b_size = 16
    sensing = (b_size / 4)
    activation = np.tanh
    W1 = np.array([.9999]).reshape((1,1))
    W2 = np.array([1.00011]).reshape((1,1))
    env = GraphEnv(agents, board_size=b_size, sensing_range=sensing)
    gcn = GCNLayer(n_nodes=agents, in_features=1, out_features=1)
    # emb = np.random.uniform(0,2, size=(agents))
    emb = np.ones(agents).reshape((agents,1))
    obs = env.reset()
    test = [200]
    for le in range(len(test)):
        obs = env.reset()
        start = time.time()
        for i in range(test[le]):
            actions = np.random.randint(0,4,size=(agents))
            emb = gcn.forward(adj_matrix=obs["adj_matrix"], node_feats=obs["embeddings"], W=W1)
            emb = gcn.forward(adj_matrix=obs["adj_matrix"], node_feats=emb, W=W2)
            # if (i+1)%50 == 0:
            #     print(f"\nEmbeddings:\n{emb}")  
            #     break
            obs, _, _, _ = env.step(actions, emb)
            # pprint(obs)
            # if le == 0:
            env.render(0,True)
            # time.sleep(0.0005)
        print(f"Simulation {test[le]} time: {time.time() - start}")
        print(f"Final embedings\n{emb}")
            

