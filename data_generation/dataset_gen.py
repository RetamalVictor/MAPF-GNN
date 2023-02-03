import sys

sys.path.append("")
import os 
import yaml
import torch
import argparse
import numpy as np
from cbs.cbs import Environment, CBS
"""
agents:
-   start: [0, 0]
    goal: [8, 8]
    name: agent0
-   start: [2, 7]
    goal: [0, 0]
    name: agent1
-   start: [6, 7]
    goal: [0, 2]
    name: agent3
map:
    dimensions: [10, 10]
    obstacles:
    - !!python/tuple [0, 1]
    - !!python/tuple [2, 1]
    - !!python/tuple [5, 5]

"""
def gen_input(dimensions: tuple[int,int], nb_obs:int, nb_agents:int) -> dict:
    
    """
        basic_agent = { 
            "start":[0,0],
            "goal":[1,1],
            "name":"agent1"
            }
    """

    input_dict = {
        "agents": [],
        "map":{
            "dimensions":dimensions,
            "obstacles":[]
            }
        }

    starts = []
    goals = []
    obstacles = []
    def assign_obstacle(obstacles):
        good = False
        while not good:
            ag_obstacle = [np.random.randint(0, dimensions[0]),np.random.randint(0, dimensions[1])]
            if ag_obstacle not in obstacles:
                good = True
        return ag_obstacle

    def assign_start(starts,obstacles):
        good = False
        while not good:
            ag_start = [np.random.randint(0, dimensions[0]),np.random.randint(0, dimensions[1])]
            if ag_start not in starts and ag_start not in obstacles:
                good = True
        return ag_start
    def assign_goal(goals, obstacles):
        good = False
        while not good:
            ag_goal = [np.random.randint(0, dimensions[0]),np.random.randint(0, dimensions[1])]
            if ag_goal not in goals and ag_goal not in obstacles:
                good = True
        return ag_goal

    for obstacle in range(nb_obs):
        obstacle = assign_obstacle(obstacles)
        obstacles.append(obstacle)
        input_dict["map"]["obstacles"].append(obstacle)

    for agent in range(nb_agents):
        start = assign_start(starts,obstacles)
        starts.append(start)
        goal = assign_goal(goals,obstacles)
        goals.append(goal)
        input_dict["agents"].append({ 
            "start":start,
            "goal":goal,
            "name":f"agent{agent}"
            })

    return input_dict
    # OBS 0 for now




def data_gen(input_dict, output_path):

    os.makedirs(output_path)
    param = input_dict
    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']

    env = Environment(dimension, agents, obstacles)

    # Searching
    cbs = CBS(env, verbose=False)
    solution = cbs.search()
    if not solution:
        print(" Solution not found" )
        return

    # Write to output file
    output = dict()
    output["schedule"] = solution
    output["cost"] = env.compute_solution_cost(solution)
    solution_path = os.path.join(output_path, "solution.yaml")
    with open(solution_path, 'w') as solution_path:
        yaml.safe_dump(output, solution_path)

    parameters_path = os.path.join(output_path, "input.yaml")
    with open(parameters_path, 'w') as parameters_path:
        yaml.safe_dump(param, parameters_path)

def create_solutions(path, num_cases, config): 
    print("Generating solutions")
    for i in range(num_cases):
        if i%25 == 0:
            print(f"Solution -- [{i}/{num_cases}]")
        inpt = gen_input(config["map_shape"],config["nb_obstacles"],config["nb_agents"])
        data_gen(inpt, os.path.join(path, f"case_{i}"))
    print(f"Cases stored in {path}")


if __name__ == "__main__":
    path = fr"dataset\4_3_8v2\train"
    config = {
        "device":"cpu",
        "num_agents":3,
        "map_shape":[8,8],
        "root_dir": path,
        "nb_agents": 4,
        "nb_obstacles": 3
        }
    create_solutions(path, 2000, config)
    path = fr"dataset\4_3_8v2\val"
    config = {
        "device":"cpu",
        "num_agents":3,
        "map_shape":[8,8],
        "root_dir": path,
        "nb_agents": 4,
        "nb_obstacles": 3
        }
    create_solutions(path, 2000, config)
    # total = 200
    # for i in range(0,total):
    #     if i%25 == 0:
    #         print(f"Solution[{i}/{total}]")
    #     inpt = gen_input([5,5],0,2)
    #     data_gen(inpt, path)
    # print(f"Solution[{i}/{total}]")    
