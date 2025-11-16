#!/usr/bin/env python3
"""
Model Evaluation Script for MAPF-GNN
Provides comprehensive evaluation with benchmarking and visualization options.
"""

import yaml
import argparse
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
from grid.env_graph_gridv1 import GraphEnv, create_goals, create_obstacles


# Terminal colors for better logging
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log_info(msg):
    """Print info message in cyan."""
    print(f"{Colors.OKCYAN}[INFO]{Colors.ENDC} {msg}")


def log_success(msg):
    """Print success message in green."""
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {msg}")


def log_warning(msg):
    """Print warning message in yellow."""
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {msg}")


def log_error(msg):
    """Print error message in red."""
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {msg}")


def log_header(msg):
    """Print header message in bold."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{msg:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate MAPF-GNN model with comprehensive metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model and configuration
    parser.add_argument("--config", type=str, default="configs/config_gnn.yaml",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model checkpoint (overrides config)")

    # Evaluation settings
    parser.add_argument("--episodes", type=int, default=None,
                       help="Number of episodes to evaluate (overrides config)")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum steps per episode (overrides config)")

    # Visualization
    parser.add_argument("--render", action="store_true", default=False,
                       help="Enable visualization during evaluation")
    parser.add_argument("--render-delay", type=float, default=0.001,
                       help="Delay between render frames (seconds)")

    # Benchmarking
    parser.add_argument("--benchmark", action="store_true", default=False,
                       help="Enable detailed performance benchmarking")
    parser.add_argument("--save-results", type=str, default=None,
                       help="Save results to JSON file")

    # Logging
    parser.add_argument("--verbose", action="store_true", default=False,
                       help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", default=False,
                       help="Suppress episode-by-episode output")

    return parser.parse_args()


class EpisodeStats:
    """Track statistics for a single episode."""
    def __init__(self):
        self.success_rate = 0.0
        self.flow_time = 0
        self.steps_taken = 0
        self.inference_time = 0.0
        self.episode_time = 0.0
        self.collisions = 0
        self.agents_at_goal = 0
        self.total_agents = 0

    def to_dict(self):
        return {
            'success_rate': float(self.success_rate),
            'flow_time': int(self.flow_time),
            'steps_taken': int(self.steps_taken),
            'inference_time_ms': float(self.inference_time * 1000),
            'episode_time_s': float(self.episode_time),
            'collisions': int(self.collisions),
            'agents_at_goal': int(self.agents_at_goal),
            'total_agents': int(self.total_agents),
        }


class Evaluator:
    """Main evaluation class for MAPF-GNN models."""

    def __init__(self, args):
        self.args = args
        self.config = self.load_config()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config["device"] = self.device

        # Load model
        self.model = self.load_model()

        # Statistics tracking
        self.episode_stats = []
        self.start_time = None

        # Setup visualization
        if args.render:
            plt.ion()
            log_info("Visualization enabled")

    def load_config(self):
        """Load configuration from YAML file."""
        log_info(f"Loading configuration from {self.args.config}")
        with open(self.args.config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Override config with command-line arguments
        if self.args.episodes is not None:
            config["tests_episodes"] = self.args.episodes
        if self.args.max_steps is not None:
            config["max_steps"] = self.args.max_steps

        return config

    def load_model(self):
        """Load the neural network model."""
        net_type = self.config["net_type"]
        msg_type = self.config.get("msg_type", "gcn")

        log_info(f"Loading {net_type} model (message type: {msg_type})")

        # Import appropriate network
        if net_type == "gnn":
            if msg_type == "message":
                from models.framework_gnn_message import Network
            else:
                from models.framework_gnn import Network
        elif net_type == "baseline":
            from models.framework_baseline import Network
        else:
            log_error(f"Unknown network type: {net_type}")
            raise ValueError(f"Unknown network type: {net_type}")

        model = Network(self.config)
        model.to(self.device)

        # Load weights
        if self.args.model:
            model_path = Path(self.args.model)
        else:
            exp_name = self.config["exp_name"]
            model_path = Path("trained_models") / exp_name / "model.pt"
            if not model_path.exists():
                model_path = Path("results") / exp_name / "model.pt"

        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            log_success(f"Loaded model from {model_path}")
        else:
            log_warning(f"No model found at {model_path}, using random weights")

        model.eval()
        return model

    def run_episode(self, episode_num):
        """Run a single evaluation episode."""
        stats = EpisodeStats()
        episode_start = time.time()

        # Create environment
        obstacles = create_obstacles(self.config["board_size"], self.config["obstacles"])
        goals = create_goals(self.config["board_size"], self.config["num_agents"], obstacles)
        env = GraphEnv(
            self.config,
            goal=goals,
            obstacles=obstacles,
            sensing_range=self.config.get("sensing_range", 6)
        )

        emb = env.getEmbedding()
        obs = env.reset()

        stats.total_agents = self.config["num_agents"]
        max_steps = self.config["max_steps"] + 10

        # Episode loop
        for step in range(max_steps):
            # Prepare inputs
            fov = torch.tensor(obs["fov"]).float().unsqueeze(0).to(self.device)
            gso = torch.tensor(obs["adj_matrix"]).float().unsqueeze(0).to(self.device)

            # Inference with timing
            inference_start = time.time()
            with torch.no_grad():
                if self.config["net_type"] == "gnn":
                    action = self.model(fov, gso)
                else:
                    action = self.model(fov)
                action = action.cpu().squeeze(0).numpy()
            stats.inference_time += time.time() - inference_start

            action = np.argmax(action, axis=1)

            # Step environment
            obs, reward, done, info = env.step(action, emb)
            stats.steps_taken = step + 1

            # Render if enabled
            if self.args.render:
                env.render(None)
                if self.args.render_delay > 0:
                    time.sleep(self.args.render_delay)

            # Check termination
            if done:
                if self.args.verbose:
                    log_success(f"Episode {episode_num+1}: All agents reached goals in {step+1} steps")
                break

            if step == max_steps - 1:
                if self.args.verbose:
                    log_warning(f"Episode {episode_num+1}: Max steps reached")

        # Compute final metrics
        metrics = env.computeMetrics()
        stats.success_rate = metrics[0]
        stats.flow_time = metrics[1]
        stats.agents_at_goal = int(stats.success_rate * stats.total_agents)
        stats.episode_time = time.time() - episode_start

        # Log episode summary
        if not self.args.quiet:
            self.log_episode(episode_num, stats)

        return stats

    def log_episode(self, episode_num, stats):
        """Log statistics for a single episode."""
        total_episodes = self.config["tests_episodes"]
        success_str = f"{stats.success_rate*100:5.1f}%"

        if stats.success_rate == 1.0:
            color = Colors.OKGREEN
        elif stats.success_rate >= 0.5:
            color = Colors.WARNING
        else:
            color = Colors.FAIL

        print(f"Episode {episode_num+1:3d}/{total_episodes}: "
              f"Success: {color}{success_str}{Colors.ENDC} | "
              f"Steps: {stats.steps_taken:3d} | "
              f"Flow Time: {stats.flow_time:4d} | "
              f"Inference: {stats.inference_time*1000:5.1f}ms")

    def evaluate(self):
        """Run full evaluation."""
        log_header("MAPF-GNN Model Evaluation")

        # Print configuration
        self.print_config()

        # Run episodes
        num_episodes = self.config["tests_episodes"]
        log_info(f"Starting evaluation with {num_episodes} episodes")
        print()

        self.start_time = time.time()

        for ep in range(num_episodes):
            stats = self.run_episode(ep)
            self.episode_stats.append(stats)

        total_time = time.time() - self.start_time

        # Print results
        self.print_results(total_time)

        # Save results if requested
        if self.args.save_results:
            self.save_results()

    def print_config(self):
        """Print evaluation configuration."""
        print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
        print(f"  Model: {self.config['exp_name']}")
        print(f"  Network Type: {self.config['net_type']}")
        print(f"  Device: {self.device}")
        print(f"  Board Size: {self.config['board_size']}")
        print(f"  Agents: {self.config['num_agents']}")
        print(f"  Obstacles: {self.config['obstacles']}")
        print(f"  Episodes: {self.config['tests_episodes']}")
        print(f"  Max Steps: {self.config['max_steps']}")
        print(f"  Render: {self.args.render}")
        print(f"  Benchmark: {self.args.benchmark}")
        print()

    def print_results(self, total_time):
        """Print final evaluation results."""
        print()
        log_header("Evaluation Results")

        # Aggregate statistics
        success_rates = [s.success_rate for s in self.episode_stats]
        flow_times = [s.flow_time for s in self.episode_stats]
        steps_taken = [s.steps_taken for s in self.episode_stats]
        inference_times = [s.inference_time for s in self.episode_stats]

        complete_success = sum(1 for s in success_rates if s == 1.0)

        # Print summary statistics
        print(f"{Colors.BOLD}Success Metrics:{Colors.ENDC}")
        print(f"  Complete Success: {complete_success}/{len(success_rates)} "
              f"({complete_success/len(success_rates)*100:.1f}%)")
        print(f"  Avg Success Rate: {np.mean(success_rates)*100:.2f}% "
              f"(±{np.std(success_rates)*100:.2f}%)")
        print()

        print(f"{Colors.BOLD}Path Metrics:{Colors.ENDC}")
        print(f"  Avg Steps Taken: {np.mean(steps_taken):.1f} (±{np.std(steps_taken):.1f})")
        print(f"  Avg Flow Time: {np.mean(flow_times):.1f} (±{np.std(flow_times):.1f})")
        print(f"  Max Flow Time: {np.max(flow_times)}")
        print()

        print(f"{Colors.BOLD}Performance Metrics:{Colors.ENDC}")
        avg_inference = np.mean(inference_times) * 1000
        total_inferences = sum(s.steps_taken for s in self.episode_stats)
        print(f"  Avg Inference Time: {avg_inference:.2f}ms per step")
        print(f"  Total Inferences: {total_inferences}")
        print(f"  Inference FPS: {1000/avg_inference:.1f} steps/second")
        print()

        print(f"{Colors.BOLD}Overall:{Colors.ENDC}")
        print(f"  Total Episodes: {len(self.episode_stats)}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Avg Time per Episode: {total_time/len(self.episode_stats):.2f}s")
        print()

        # Benchmark details
        if self.args.benchmark:
            self.print_benchmark_details()

    def print_benchmark_details(self):
        """Print detailed benchmarking information."""
        log_header("Benchmark Details")

        print(f"{Colors.BOLD}Per-Episode Statistics:{Colors.ENDC}\n")

        # Sort episodes by success rate
        sorted_stats = sorted(enumerate(self.episode_stats),
                            key=lambda x: x[1].success_rate,
                            reverse=True)

        print(f"{'Ep':<4} {'Success':<8} {'Steps':<6} {'Flow':<6} {'Inf(ms)':<8} {'Time(s)':<8}")
        print("-" * 50)

        for ep_num, stats in sorted_stats[:10]:  # Top 10
            print(f"{ep_num+1:<4} {stats.success_rate*100:>6.1f}% {stats.steps_taken:>6} "
                  f"{stats.flow_time:>6} {stats.inference_time*1000:>8.2f} "
                  f"{stats.episode_time:>8.2f}")

        if len(sorted_stats) > 10:
            print(f"... ({len(sorted_stats)-10} more episodes)")
        print()

    def save_results(self):
        """Save evaluation results to JSON file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model': self.config['exp_name'],
                'net_type': self.config['net_type'],
                'board_size': self.config['board_size'],
                'num_agents': self.config['num_agents'],
                'obstacles': self.config['obstacles'],
                'episodes': self.config['tests_episodes'],
            },
            'summary': {
                'complete_success': sum(1 for s in self.episode_stats if s.success_rate == 1.0),
                'avg_success_rate': float(np.mean([s.success_rate for s in self.episode_stats])),
                'std_success_rate': float(np.std([s.success_rate for s in self.episode_stats])),
                'avg_flow_time': float(np.mean([s.flow_time for s in self.episode_stats])),
                'avg_steps': float(np.mean([s.steps_taken for s in self.episode_stats])),
                'avg_inference_ms': float(np.mean([s.inference_time for s in self.episode_stats]) * 1000),
            },
            'episodes': [stats.to_dict() for stats in self.episode_stats]
        }

        output_path = Path(self.args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        log_success(f"Results saved to {output_path}")


def main():
    """Main entry point."""
    args = parse_args()

    try:
        evaluator = Evaluator(args)
        evaluator.evaluate()
    except KeyboardInterrupt:
        log_warning("\nEvaluation interrupted by user")
    except Exception as e:
        log_error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
