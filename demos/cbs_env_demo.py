"""
Simple demo: Solve MAPF with CBS and visualize in GraphEnv using render().

author: Victor Retamal
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import time
from cbs.cbs import Environment, CBS
from grid.env_graph_gridv1 import GraphEnv


def cbs_to_actions(cbs_solution):
    """Convert CBS solution to action sequence."""
    if not cbs_solution:
        return None

    n_agents = len(cbs_solution)
    max_steps = max(len(path) for path in cbs_solution.values()) - 1
    actions = np.zeros((max_steps, n_agents), dtype=np.int32)

    # Action mapping
    for agent_idx, (_, path) in enumerate(cbs_solution.items()):
        for t in range(len(path) - 1):
            dx = path[t+1]["x"] - path[t]["x"]
            dy = path[t+1]["y"] - path[t]["y"]

            if dx == 1: actions[t, agent_idx] = 1  # Right
            elif dx == -1: actions[t, agent_idx] = 3  # Left
            elif dy == 1: actions[t, agent_idx] = 2  # Up
            elif dy == -1: actions[t, agent_idx] = 4  # Down
            else: actions[t, agent_idx] = 0  # Wait

    return actions


def main():
    # Simple test case
    board_size = [10, 10]

    # Define agents and obstacles
    agents = [
        {"name": "agent0", "start": [1, 1], "goal": [8, 8]},
        {"name": "agent1", "start": [8, 1], "goal": [1, 8]},
        {"name": "agent2", "start": [1, 8], "goal": [8, 1]},
    ]

    obstacles = [(5, 5), (4, 5), (5, 4), (6, 5), (5, 6)]  # Plus shape

    print("="*60)
    print("SOLVING WITH CBS")
    print("="*60)

    # Solve with CBS
    cbs_env = Environment(board_size, agents, obstacles)
    cbs = CBS(cbs_env, verbose=False)
    solution = cbs.search()

    if not solution:
        print("No solution found!")
        return

    print(f"✅ Solution found! Total cost: {sum(len(p) for p in solution.values())}")

    # Convert to actions
    action_sequence = cbs_to_actions(solution)
    n_steps, n_agents = action_sequence.shape

    print(f"Steps: {n_steps}, Agents: {n_agents}")

    # Create GraphEnv
    config = {
        "num_agents": n_agents,
        "board_size": board_size,
        "max_time": n_steps + 10,
        "min_time": n_steps // 2,
    }

    # Extract goals as numpy array
    goals = np.array([a["goal"] for a in agents])
    obstacles_array = np.array(obstacles) if obstacles else np.array([]).reshape(0, 2)

    # Create environment
    env = GraphEnv(config, goal=goals, obstacles=obstacles_array)
    env.reset()

    # Set starting positions
    for i, agent in enumerate(agents):
        env.positionX[i] = agent["start"][0]
        env.positionY[i] = agent["start"][1]

    print("\n" + "="*60)
    print("PLAYING SOLUTION IN ENVIRONMENT")
    print("="*60)

    # Setup interactive plotting
    plt.ion()
    fig = plt.figure(figsize=(10, 10))

    # Play the solution
    for step in range(n_steps):
        # Clear previous plot
        plt.clf()

        # Use environment's built-in render - now it won't auto-clear
        env.render(agentId=None, mode="plot")

        plt.title(f"Step {step+1}/{n_steps}")
        plt.draw()
        plt.pause(0.5)  # Control the display timing ourselves

        # Execute action
        actions = action_sequence[step]
        embeddings = np.ones((n_agents, 1))  # Dummy embeddings

        obs, rewards, dones, info = env.step(actions, embeddings)

        # Print status
        print(f"Step {step+1}: ", end="")
        for i in range(n_agents):
            print(f"A{i}:({env.positionX[i]},{env.positionY[i]}) ", end="")
        # rewards is a dict, sum the values
        total_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards.sum()
        print(f"| Rewards: {total_reward:.2f}")

        if np.all(dones):
            print("\n✅ All agents reached their goals!")
            break

    # Final render
    plt.clf()
    env.render(agentId=None, mode="plot")
    plt.title("Final State - All Agents Completed")
    plt.draw()

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()