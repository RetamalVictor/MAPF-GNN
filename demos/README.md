# MAPF-GNN Demos

## CBS Environment Demo (`cbs_env_demo.py`)

This demo demonstrates how to:
1. Solve a Multi-Agent Path Finding (MAPF) problem using Conflict-Based Search (CBS)
2. Convert the CBS solution to action sequences
3. Execute the solution in the GraphEnv environment
4. Visualize the agents navigating to their goals

### Running the demo

```bash
cd demos
uv run python cbs_env_demo.py
```

### What it does

- Creates a 10x10 grid with 3 agents and obstacles in a plus-shape pattern
- Uses CBS to find optimal collision-free paths for all agents
- Executes the solution step-by-step in the GraphEnv environment
- Shows a visualization of agents (colored dots) moving to their goals (blue stars) while avoiding obstacles (black squares)

The demo confirms that:
- The CBS algorithm finds optimal solutions
- The GraphEnv environment correctly executes action sequences
- The fixed `render()` method properly displays the environment state