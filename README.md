# Decentralized Multi-Agent Path Planning with Graph Neural Networks

[![CI](https://github.com/RetamalVictor/MAPF-GNN/actions/workflows/ci.yml/badge.svg)](https://github.com/RetamalVictor/MAPF-GNN/actions/workflows/ci.yml)

A Graph Neural Network (GNN) based approach for decentralized multi-robot path planning. This implementation replicates and extends the paper **"Graph Neural Networks for Decentralized Multi-Robot Path Planning"** by Qingbiao Li et al.

|            Multi-Agent Path Finding in Action            |
|:--------------------------------------------------------:|
![Demo](example.gif)

**Presentation Slides:** [Google Slides](https://docs.google.com/presentation/d/1U5GJXuAFZTgo84--idJMGrxTX976u6J98tHx4gF_Jyw/edit?usp=sharing)

---

## Features

- **Decentralized Planning**: Agents make independent decisions using only local observations
- **GNN Architecture**: Graph Convolutional Networks with spectral filters and message passing
- **Conflict-Based Search (CBS)**: Optimal path planning algorithm for dataset generation
- **Gymnasium Environment**: Custom grid-based environment for multi-agent navigation
- **Comprehensive Testing**: 44 unit tests covering CBS, environment, and performance
- **Visualization Tools**: Real-time rendering of agent movements and trajectories
- **Benchmarking Suite**: Performance profiling for scalability analysis

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Generation](#dataset-generation)
- [Training](#training)
- [Inference & Evaluation](#inference--evaluation)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Contributing](#contributing)

---

## Installation

### Prerequisites

- Python 3.10 or 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/RetamalVictor/MAPF-GNN.git
cd MAPF-GNN
```

2. **Install dependencies**

Using `uv` (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Or using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Start

### Run a Demo

See GNN-based multi-agent planning in action:

```bash
uv run python example.py
```

This runs a demonstration with:
- 5 agents on a 16×16 grid
- 8 obstacles
- GCN-based policy network
- Real-time visualization

### Visualize CBS Algorithm

Watch the Conflict-Based Search algorithm solve a MAPF problem:

```bash
uv run python demos/cbs_env_demo.py
```

This demonstrates:
- CBS finding optimal collision-free paths
- Action sequence execution in the environment
- Step-by-step visualization of agent movements

---

## Dataset Generation

The training data consists of expert trajectories generated using the Conflict-Based Search (CBS) algorithm.

### Generate a Dataset

```bash
cd data_generation
uv run python main_data.py
```

**Default configuration:**
- 5 agents
- 28×28 grid
- 8 obstacles
- 1000 trajectory samples
- Output: `dataset/5_8_28/`

### Customize Dataset Parameters

Edit `data_generation/main_data.py` to modify:

```python
# Grid configuration
board_size = [28, 28]      # Grid dimensions
num_agents = 5             # Number of agents
num_obstacles = 8          # Number of obstacles
num_samples = 1000         # Training samples to generate

# Agent parameters
max_time = 32              # Maximum timesteps per episode
sensing_range = 6          # Agent's observation radius
```

### Dataset Structure

Generated datasets are organized as:

```
dataset/
└── 5_8_28/                    # {agents}_{obstacles}_{grid_size}
    ├── train/
    │   ├── trajectory_0000.pkl
    │   ├── trajectory_0001.pkl
    │   └── ...
    └── metadata.json          # Dataset statistics
```

Each trajectory file contains:
- Initial positions and goals for all agents
- Optimal paths computed by CBS
- Local observations at each timestep
- Actions taken by each agent

### Visualize Dataset Samples

To visualize trajectories from your dataset:

```python
from data_generation.trajectory_parser import TrajectoryParser
from grid.env_graph_gridv1 import GraphEnv
import matplotlib.pyplot as plt

# Load a trajectory
parser = TrajectoryParser('dataset/5_8_28/train')
trajectory = parser.load_trajectory(0)

# Create environment and visualize
env = GraphEnv(config={
    'board_size': [28, 28],
    'num_agents': 5,
    # ... other config
})
env.reset()
env.render()
plt.show()
```

---

## Training

Train a GNN model for decentralized path planning:

### Basic Training

```bash
uv run python train.py --config configs/config_gnn.yaml
```

### Configuration Options

The configuration file controls all training parameters:

```yaml
# configs/config_gnn.yaml

# Network Architecture
channels: [16, 16, 16]          # CNN encoder channels
encoder_dims: [64]              # Encoder hidden dimensions
node_dims: [128]                # GNN node embedding dimensions
graph_filters: [3]              # GCN filter sizes (K-hop aggregation)

# Training
epochs: 50                      # Number of training epochs
batch_size: 128                 # Batch size
learning_rate: 0.001            # Adam learning rate

# Environment
board_size: [18, 18]            # Grid size for training
num_agents: 5                   # Number of agents
obstacles: 5                    # Number of obstacles
max_steps: 32                   # Maximum episode length
sensing_range: 6                # Agent observation radius

# Dataset
train:
    root_dir: 'dataset/5_8_28'  # Path to training data
    mode: 'train'
    min_time: 5
    max_time_dl: 25
```

### Available Models

- **GNN (GCN)**: Graph Convolutional Network with spectral filters
- **Baseline**: CNN encoder + MLP policy (no graph structure)

Specify the model type in config:
```yaml
net_type: 'gnn'      # Options: 'gnn', 'baseline'
msg_type: 'gcn'      # GNN aggregation: 'gcn', 'self_importance'
```

### Monitor Training

Training logs include:
- Success rate (agents reaching goals)
- Collision rate
- Average path length
- Training loss

Models are saved to `trained_models/{exp_name}/`

---

## Inference & Evaluation

### Run Inference on Trained Model

Use one of the pretrained models:

```bash
# Using GNN with 3-hop filters (recommended)
uv run python example.py --config configs/config_gnn.yaml --model trained_models/gnn_k3/model.pt

# Or try other models:
# uv run python example.py --model trained_models/baseline/model.pt
# uv run python example.py --model trained_models/gnn_k2/model.pt
# uv run python example.py --model trained_models/gnn_msg_k3/model.pt
```

**Available pretrained models:**
- `baseline/model.pt` - CNN + MLP baseline (138KB)
- `gnn_k2/model.pt` - GNN with 2-hop filters (193KB)
- `gnn_k3/model.pt` - GNN with 3-hop filters (225KB)
- `gnn_msg_k3/model.pt` - GNN with message passing (257KB)

### Evaluate Model Performance

Test your trained model on a held-out test set:

```python
from models.framework_gnn import GNNFramework
from data_loader import load_test_data
import torch

# Load model
model = GNNFramework(config)
model.load_state_dict(torch.load('trained_models/gnn_k3/model.pt'))
model.eval()

# Load test data
test_loader = load_test_data('dataset/5_8_28/test')

# Evaluate
metrics = model.evaluate(test_loader, num_episodes=100)
print(f"Success Rate: {metrics['success_rate']:.2%}")
print(f"Collision Rate: {metrics['collision_rate']:.2%}")
print(f"Avg Path Length: {metrics['avg_path_length']:.1f}")
```

---

## Project Structure

```
MAPF-GNN/
├── cbs/                       # Conflict-Based Search implementation
│   ├── cbs.py                # CBS high-level search
│   └── a_star.py             # A* low-level planner
│
├── data_generation/           # Dataset creation tools
│   ├── main_data.py          # Dataset generation script
│   ├── dataset_gen.py        # CBS trajectory generator
│   ├── trajectory_parser.py  # Trajectory data parser
│   └── record.py             # Environment recorder
│
├── grid/                      # Gymnasium environment
│   └── env_graph_gridv1.py   # Multi-agent grid environment
│
├── models/                    # Neural network architectures
│   ├── framework_gnn.py      # GNN training framework
│   └── networks/
│       └── gnn.py            # GCN and message passing layers
│
├── configs/                   # Training configurations
│   ├── config_gnn.yaml       # GNN configuration
│   └── config_baseline.yaml  # Baseline configuration
│
├── tests/                     # Unit tests
│   ├── test_cbs.py           # CBS algorithm tests
│   ├── test_env.py           # Environment tests
│   └── test_cbs_performance.py # Performance tests
│
├── benchmarks/                # Performance benchmarks
│   ├── benchmark_performance.py
│   └── benchmark_scaling.py
│
├── demos/                     # Demo scripts
│   └── cbs_env_demo.py       # CBS visualization demo
│
├── trained_models/            # Saved model checkpoints
├── train.py                   # Training script
├── example.py                 # Quick demo script
└── data_loader.py            # Dataset loading utilities
```

---

## Testing

Run the comprehensive test suite:

### Run All Tests

```bash
uv run pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Test CBS algorithm
uv run pytest tests/test_cbs.py -v

# Test environment functionality
uv run pytest tests/test_env.py -v

# Test performance optimizations
uv run pytest tests/test_cbs_performance.py -v
```

### Test Coverage

The test suite includes:

- **CBS Algorithm Tests** (24 tests)
  - Path finding correctness
  - Conflict detection and resolution
  - Edge case handling
  - Heap-based optimization verification

- **Environment Tests** (17 tests)
  - Agent movement and collision detection
  - Obstacle generation and handling
  - Field of view computation
  - Success rate calculation

- **Performance Tests** (3 tests)
  - Scalability benchmarks
  - Memory efficiency tests
  - Optimization validation

**Current Status:** ✅ All 44 tests passing

---

## Benchmarks

Profile system performance across different scenarios:

### CBS Performance Benchmark

```bash
uv run python benchmarks/benchmark_performance.py
```

Measures CBS solver performance for:
- Different numbers of agents (2-10)
- Varying grid sizes (10×10 to 50×50)
- Different obstacle densities

### Scalability Benchmark

```bash
uv run python benchmarks/benchmark_scaling.py
```

Tests how performance scales with:
- Number of agents
- Grid complexity
- Search depth

### Sample Output

```
CBS Performance Benchmark Results:
=====================================
Agents: 5, Grid: 20×20, Obstacles: 15
  - Solution found: ✓
  - Time: 124.3ms
  - Nodes expanded: 187
  - Path optimality: 1.0

Agents: 10, Grid: 30×30, Obstacles: 25
  - Solution found: ✓
  - Time: 892.5ms
  - Nodes expanded: 1,423
  - Path optimality: 1.0
```

---

## Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `uv pip install pytest`
4. Make your changes
5. Run tests: `uv run pytest tests/`
6. Commit changes: `git commit -am 'Add feature'`
7. Push branch: `git push origin feature/my-feature`
8. Create a Pull Request

### Code Quality

- All new code should include tests
- Tests must pass before merging
- Follow existing code style and conventions
- Add docstrings for public functions

### Continuous Integration

GitHub Actions automatically:
- Runs all tests on push/PR
- Tests on Python 3.10 and 3.11
- Validates code quality

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2020graph,
  title={Graph neural networks for decentralized multi-robot path planning},
  author={Li, Qingbiao and Gama, Fernando and Ribeiro, Alejandro and Prorok, Amanda},
  journal={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
}
```

---

## License

This project is part of a replication study for the course "Machine Learning for Graphs" @ VU Amsterdam.

---

## Acknowledgments

- Original paper authors: Qingbiao Li, Fernando Gama, Alejandro Ribeiro, Amanda Prorok
- VU Amsterdam Machine Learning for Graphs course
- Conflict-Based Search algorithm by Sharon et al.

---

## Contact

For questions or issues, please [open an issue](https://github.com/RetamalVictor/MAPF-GNN/issues) on GitHub.
