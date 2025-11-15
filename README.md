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

### Run a Simple Demo

See GNN-based multi-agent planning in action with a single episode:

```bash
uv run python example.py
```

This runs a single demonstration episode with:
- Real-time visualization
- Step-by-step agent movement
- Success metrics displayed

### Evaluate Model Performance

Run comprehensive evaluation with detailed statistics:

```bash
# Quick evaluation (no visualization, 100 episodes from config)
uv run python evaluate.py

# With visualization
uv run python evaluate.py --render

# Custom episodes with benchmarking
uv run python evaluate.py --episodes 50 --benchmark

# Save results to JSON
uv run python evaluate.py --episodes 100 --save-results results/evaluation.json
```

**Key features:**
- Color-coded terminal output
- Success rate, flow time, and performance metrics
- Optional visualization (--render)
- Benchmarking mode with detailed per-episode stats
- JSON export for analysis

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

### Professional Evaluation Tool

The `evaluate.py` script provides comprehensive model evaluation with detailed metrics:

```bash
# Basic evaluation
uv run python evaluate.py

# With all options
uv run python evaluate.py \
  --config configs/config_gnn.yaml \
  --model trained_models/gnn_k3/model.pt \
  --episodes 100 \
  --render \
  --benchmark \
  --save-results results/eval_$(date +%Y%m%d).json
```

**Command-Line Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--config PATH` | Configuration file | `configs/config_gnn.yaml` |
| `--model PATH` | Model checkpoint | From config |
| `--episodes N` | Number of test episodes | From config (100) |
| `--max-steps N` | Max steps per episode | From config |
| `--render` | Enable visualization | Off |
| `--render-delay SECONDS` | Delay between frames | 0.001 |
| `--benchmark` | Detailed performance stats | Off |
| `--save-results PATH` | Save to JSON file | None |
| `--verbose` | Detailed logging | Off |
| `--quiet` | Minimal output | Off |

**Output Metrics:**
- Complete success rate (all agents reach goals)
- Average success rate per episode
- Average and max flow time
- Average steps taken
- Inference time and FPS
- Per-episode statistics (in benchmark mode)

**Example Output:**
```
============================================================
                  MAPF-GNN Model Evaluation
============================================================

Configuration:
  Model: gnn_k3
  Network Type: gnn
  Device: cpu
  Board Size: [18, 18]
  Agents: 5
  Obstacles: 5
  Episodes: 100

Episode   1/100: Success:  100.0% | Steps:  24 | Flow Time:  120 | Inference:   8.2ms
Episode   2/100: Success:  100.0% | Steps:  28 | Flow Time:  140 | Inference:   9.1ms
...

============================================================
                    Evaluation Results
============================================================

Success Metrics:
  Complete Success: 87/100 (87.0%)
  Avg Success Rate: 94.20% (±12.30%)

Path Metrics:
  Avg Steps Taken: 26.4 (±5.2)
  Avg Flow Time: 132.1 (±26.0)
  Max Flow Time: 189

Performance Metrics:
  Avg Inference Time: 8.45ms per step
  Total Inferences: 2640
  Inference FPS: 118.3 steps/second
```

### Available Pretrained Models

**Models:**
- `baseline/model.pt` - CNN + MLP baseline (138KB)
- `gnn_k2/model.pt` - GNN with 2-hop filters (193KB)
- `gnn_k3/model.pt` - GNN with 3-hop filters (225KB) **← Recommended**
- `gnn_msg_k3/model.pt` - GNN with message passing (257KB)

**Quick Model Comparison:**
```bash
# Evaluate all models
for model in baseline gnn_k2 gnn_k3 gnn_msg_k3; do
  echo "Evaluating $model..."
  uv run python evaluate.py \
    --model trained_models/$model/model.pt \
    --episodes 50 \
    --quiet \
    --save-results results/${model}_eval.json
done
```

### Simple Demo

For a quick visual demonstration with a single episode:

```bash
uv run python example.py
```

This shows one complete episode with visualization and basic metrics.

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
│   ├── test_cbs_performance.py # Performance tests
│   └── test_data_pipeline.py # Data generation tests
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
├── evaluate.py                # Professional evaluation tool
├── example.py                 # Quick demo script
└── data_loader.py            # Dataset loading utilities
```

---


## Contact

For questions or issues, please [open an issue](https://github.com/RetamalVictor/MAPF-GNN/issues) on GitHub.
