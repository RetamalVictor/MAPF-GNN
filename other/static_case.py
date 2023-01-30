import numpy as np
from dotsEnv import DotsEnv
from pprint import pprint

if __name__ == '__main__':
    env = DotsEnv(nb_agents=3, target_random_start=False, active_colisions=False, change_target=True)
    env.reset()
    for i in range(10):
        action = {str(i): [np.random.uniform(-1,1),np.random.uniform(-1,1)] for i in range(3)}
        obs, r, dones, info = env.step(action)
        pprint(obs["0"])
        env.render()

def static_case_control(obs):
    