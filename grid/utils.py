import numpy as np

def create_goals(board_size, num_agents, obstacles=None):
    avilable_pos_x = np.arange(board_size[0])
    avilable_pos_y = np.arange(board_size[1])
    if obstacles is not None:
        mask_x = np.isin(avilable_pos_x, obstacles[:, 0])
        mask_y = np.isin(avilable_pos_y, obstacles[:, 1])
        avilable_pos_x = avilable_pos_x[~mask_x]
        avilable_pos_y = avilable_pos_y[~mask_y]
    goals_x = np.random.choice(avilable_pos_x, size=num_agents, replace=False)
    goals_y = np.random.choice(avilable_pos_y, size=num_agents, replace=False)
    goals = np.array([goals_x, goals_y]).T
    return goals


def create_obstacles(board_size, nb_obstacles):
    avilable_pos = np.arange(board_size[0])
    obstacles_x = np.random.choice(avilable_pos, size=nb_obstacles, replace=False)
    obstacles_y = np.random.choice(avilable_pos, size=nb_obstacles, replace=False)
    obstacles = np.array([obstacles_x, obstacles_y]).T
    return obstacles