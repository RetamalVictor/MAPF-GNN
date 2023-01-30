from copy import copy
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import time
from pprint import pprint
from grid.env_graph_gridv1 import GraphEnv


def show_graph(posx, posy, adjacency_matrix):
    embedding = np.linspace(0.0, 2.0, 10)
    norm = colors.Normalize(vmin=1.0, vmax=2.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)
    # color = [col for col in embedding]
    column = np.where(adjacency_matrix[0])
    column = column[0]
    print(column)
    for i in range(len(column)):
        plt.plot([posx[0], posx[column[i]]], [posy[0], posy[column[i]]], color="red")
    plt.scatter(posx, posy, s=100,color=mapper.to_rgba(embedding) )
    plt.axis("off")
    plt.show()

def show_graph_neigh(posx, posy, adjacency_matrix):
    embedding = np.linspace(0.0, 2.0, 30)
    norm = colors.Normalize(vmin=1.0, vmax=2.0, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)
    # color = [col for col in embedding]
    column = np.where(adjacency_matrix[0])
    column = column[0]
    print(column)
    for i in range(len(column)):
        plt.plot([posx[0], posx[column[i]]], [posy[0], posy[column[i]]], color="red")
        neigh =np.where(adjacency_matrix[column[i]])[0]
        for j in range(len(neigh)):
            plt.plot([posx[column[i]], posx[neigh[j]]], [posy[column[i]], posy[neigh[j]]], color="red", ls="--")

    plt.scatter(posx, posy, s=100,color=mapper.to_rgba(embedding) )
    plt.axis("off")
    plt.show()


def _computeClosest(self, d_ij):
    """
    Compute the closest neighbours of each agent
    and return the adjacency matrix
    """
    adj_matrix = np.zeros((self.nb_agents, self.nb_agents))
    for i in range(self.nb_agents):
        closest = np.argsort(d_ij[i])[:4]
        adj_matrix[i][closest] = 1
    return adj_matrix

if __name__ == "__main__":
    board_size=10
    sensing = 4
    trayectory = np.load(fr"case_0\trayectory_case_0.npy", allow_pickle=True)
    agents = trayectory.shape[0]
    start = np.load(fr"case_0\start_case_0.npy", allow_pickle=True)
    env = GraphEnv(agents, board_size=board_size, sensing_range=sensing, starting_positions=start)

    # goals = [[1,1],[1,3]]
    # starts = [[15,15],[14,15]]
    # plan = Planner.plan(starts=starts, goals=goals)
    # pprint(plan)
    map_act={
        0:"idle",
        1:"right",
        2:"up",
        3:"down",
        4:"left",
    }
    emb = np.ones(agents).reshape((agents,1))
    obs = env.reset()
    for i in range(len(trayectory[0])):
        # actions = np.random.randint(0,4,size=(agents))
        actions = trayectory[:,i]
        print(map_act[actions[1]])
        obs, _, _, _ = env.step(actions,emb)
        env.render(agentId=0, printNeigh=True)
        time.sleep(0.5)
