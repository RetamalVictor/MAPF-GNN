import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pprint import pprint

def get_longest_path(schedule):
    longest=0
    for agent in schedule.keys():
        if len(schedule[agent]) > longest:
            longest = len(schedule[agent])
    return longest

def parse_trayectories(schedule):
    longest = get_longest_path(schedule)
    trayect = np.zeros((len(schedule),longest),dtype=np.int32)
    action_map = {
        str((0,0)):0,
        str((1,0)):1,
        str((0,1)):2,
        str((-1,0)):3,
        str((0,-1)):4,
    }
    j = 0
    startings = np.zeros((len(schedule),2),dtype=np.int32)
    for agent in schedule.keys():
        prev_x = schedule[agent][0]["x"]
        prev_y = schedule[agent][0]["y"]
        startings[j][0]= prev_x
        startings[j][1]= prev_y
        
        for i in range(len(schedule[agent])):
            next_x = schedule[agent][i]["x"]
            next_y = schedule[agent][i]["y"] 
            trayect[j][i] = action_map[str((next_x - prev_x, next_y - prev_y))]
            prev_x = next_x
            prev_y = next_y
        j+=1
    return trayect, np.array(startings)

def parse_traject(path):
    cases = os.listdir(path)
    print("Parsing trayectories")
    for i in range(len(cases)):        
        with open(os.path.join(path,fr"case_{i}\solution.yaml")) as states_file:
            schedule = yaml.load(states_file, Loader=yaml.FullLoader)
        
        combined_schedule = {}
        combined_schedule.update(schedule["schedule"])

        t, s = parse_trayectories(combined_schedule)
        np.save(os.path.join(path,fr"case_{i}\trajectory.npy"), t)
        if i%25 == 0:
            print(f"Trajectoty -- [{i}/{len(cases)}]")
    print(f"Trayectoty -- [{i}/{len(cases)}]")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("schedule",type=str, default=r".\test_2.yaml", help="schedule for agents")
    # parser.add_argument("case",type=str, default="0", help="Case number")
    # args = parser.parse_args()

    # with open(args.schedule) as states_file:
    #     schedule = yaml.load(states_file, Loader=yaml.FullLoader)
    path = fr"dataset\2_0_8\train"
    cases=200
    config = {}
    parse_traject(path)


    # total = 200
    # for i in range(0,total):
    #     path = fr"dataset\2_0_5\val\case_{i}"
        
    #     with open(os.path.join(path,"solution.yaml")) as states_file:
    #         schedule = yaml.load(states_file, Loader=yaml.FullLoader)
        
    #     combined_schedule = {}
    #     combined_schedule.update(schedule["schedule"])
    #     t, s = parse_trayectories(combined_schedule)
    #     np.save(os.path.join(path,f"trajectory.npy"), t)
    #     if i%25 == 0:
    #         print(f"Trajectoty [{i}/{total}]")
    # print(f"Trayectoty [{i}/{total}]")
