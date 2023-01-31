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

    def assign_start(starts):
        good = False
        while not good:
            ag_start = [np.random.randint(0, dimensions[0]),np.random.randint(0, dimensions[1])]
            if ag_start not in starts:
                good = True
        return ag_start
    def assign_goal(goals):
        good = False
        while not good:
            ag_goal = [np.random.randint(0, dimensions[0]),np.random.randint(0, dimensions[1])]
            if ag_goal not in goals:
                good = True
        return ag_goal

    for agent in range(nb_agents):
        start = assign_start(starts)
        starts.append(start)
        goal = assign_goal(goals)
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



if __name__ == "__main__":
    total = 100
    for i in range(total):
        path = fr"dataset\2_0_5\case_{i}"
        if i%25 == 0:
            print(f"Solution[{i}/{total}]")
        inpt = gen_input([5,5],0,2)
        data_gen(inpt, path)
    print(f"Solution[{i}/{total}]")    
    print("Cases stored in dataset/")
