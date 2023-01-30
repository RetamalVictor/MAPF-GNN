from pprint import pprint
import numpy as np
from models.gnn import GCNLayer

from env_graph import GraphEnv
import time

if __name__ == "__main__":
    agents = 20
    activation = np.tanh
    W1 = np.array([0.99]).reshape((1,1))
    W2 = np.array([1.01]).reshape((1,1))
    env = GraphEnv(agents)
    gcn = GCNLayer(n_nodes=agents, in_features=1, out_features=1)
    # emb = np.random.uniform(0,2, size=(agents))
    emb = np.ones(agents).reshape((agents,1))
    obs = env.reset()
    for _ in range(3):
        obs = env.reset()
        for i in range(300):
            actions = {
                "vx": np.random.uniform(-1.1, 1.1, size=(agents)),
                "vy": np.random.uniform(-1.1, 1.1, size=(agents)),
                "headings": np.random.uniform(-3.14, 3.14, size=(agents))
            }
            emb = gcn.forward(adj_matrix=obs["adj_matrix"], node_feats=obs["embeddings"], W=W1)
            emb = gcn.forward(adj_matrix=obs["adj_matrix"], node_feats=emb, W=W2)
            if i%100 == 0:
                print(f"\nEmbeddings:\n{emb}")  
            obs, _, _, _ = env.step(actions, emb)
            # pprint(obs)
            env.render(0,True)
            time.sleep(0.05)
            

