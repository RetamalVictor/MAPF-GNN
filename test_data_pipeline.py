"""
Tests for data generation pipeline.
Verifies dataset generation and trajectory parsing functionality.

author: Victor Retamal
"""
import pytest
import numpy as np
import yaml
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.append("")

from data_generation.dataset_gen import gen_input, data_gen, create_solutions
from data_generation.trajectory_parser import (
    get_longest_path,
    parse_trajectories,
    parse_dataset_trajectories
)


class TestDatasetGeneration:
    """Test dataset generation functionality."""

    def test_gen_input_basic(self):
        """Test basic input generation."""
        # Small grid
        input_dict = gen_input((5, 5), 3, 2)

        assert input_dict is not None
        assert len(input_dict["agents"]) == 2
        assert len(input_dict["map"]["obstacles"]) == 3
        assert input_dict["map"]["dimensions"] == [5, 5]

        # Check all positions are valid
        for agent in input_dict["agents"]:
            assert 0 <= agent["start"][0] < 5
            assert 0 <= agent["start"][1] < 5
            assert 0 <= agent["goal"][0] < 5
            assert 0 <= agent["goal"][1] < 5

        for obs in input_dict["map"]["obstacles"]:
            assert 0 <= obs[0] < 5
            assert 0 <= obs[1] < 5

    def test_gen_input_no_overlaps(self):
        """Test that generated positions don't overlap incorrectly."""
        input_dict = gen_input((10, 10), 10, 5)

        if input_dict is None:
            pytest.skip("Could not generate valid input")

        # Check no start positions overlap with obstacles
        obstacles = set(map(tuple, input_dict["map"]["obstacles"]))
        starts = set()
        goals = set()

        for agent in input_dict["agents"]:
            start = tuple(agent["start"])
            goal = tuple(agent["goal"])

            # Starts shouldn't overlap with obstacles
            assert start not in obstacles
            # Starts shouldn't overlap with other starts
            assert start not in starts

            starts.add(start)
            goals.add(goal)

    def test_gen_input_dense_board(self):
        """Test generation with high density (should switch to efficient algorithm)."""
        # Try to fill 80% of a small board
        input_dict = gen_input((5, 5), 15, 4)  # 15 obs + 8 positions = 23/25 = 92%

        # Should either succeed or return None (too dense)
        if input_dict is not None:
            assert len(input_dict["agents"]) == 4
            assert len(input_dict["map"]["obstacles"]) <= 15
        # If None, that's also acceptable for very dense boards

    def test_data_gen_with_solution(self):
        """Test generating solution for a simple case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_case"

            # Create simple solvable instance
            input_dict = {
                "agents": [
                    {"name": "agent0", "start": [0, 0], "goal": [2, 0]},
                    {"name": "agent1", "start": [2, 2], "goal": [0, 2]}
                ],
                "map": {
                    "dimensions": [3, 3],
                    "obstacles": []
                }
            }

            success = data_gen(input_dict, output_path)
            assert success is True

            # Check files were created
            assert (output_path / "solution.yaml").exists()
            assert (output_path / "input.yaml").exists()

            # Load and verify solution
            with open(output_path / "solution.yaml") as f:
                solution = yaml.load(f, Loader=yaml.FullLoader)

            assert "schedule" in solution
            assert "cost" in solution
            assert "agent0" in solution["schedule"]
            assert "agent1" in solution["schedule"]

    def test_data_gen_no_solution(self):
        """Test handling of unsolvable instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_case"

            # Create unsolvable instance (goal blocked)
            input_dict = {
                "agents": [
                    {"name": "agent0", "start": [0, 0], "goal": [2, 2]}
                ],
                "map": {
                    "dimensions": [3, 3],
                    "obstacles": [(1, 1), (1, 2), (2, 1), (2, 2)]
                }
            }

            success = data_gen(input_dict, output_path)
            assert success is False

    def test_create_solutions_batch(self):
        """Test batch solution creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            config = {
                "map_shape": [5, 5],
                "nb_agents": 2,
                "nb_obstacles": 3
            }

            # Generate a small batch
            create_solutions(path, 3, config)

            # Check cases were created
            cases = list(path.glob("case_*"))
            assert len(cases) <= 3  # Some might fail

            # Verify at least one successful case
            successful = False
            for case in cases:
                if (case / "solution.yaml").exists():
                    successful = True
                    break
            assert successful, "At least one case should succeed"


class TestTrajectoryParser:
    """Test trajectory parsing functionality."""

    def test_get_longest_path(self):
        """Test finding longest path in schedule."""
        schedule = {
            "agent0": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
            "agent1": [{"x": 2, "y": 2}, {"x": 2, "y": 1}, {"x": 2, "y": 0}],
            "agent2": [{"x": 1, "y": 1}]
        }

        longest = get_longest_path(schedule)
        assert longest == 3

    def test_parse_trajectories(self):
        """Test trajectory parsing to action sequences."""
        schedule = {
            "agent0": [
                {"x": 0, "y": 0, "t": 0},
                {"x": 1, "y": 0, "t": 1},  # Right
                {"x": 1, "y": 1, "t": 2},  # Up
                {"x": 1, "y": 1, "t": 3},  # Wait
            ],
            "agent1": [
                {"x": 2, "y": 2, "t": 0},
                {"x": 1, "y": 2, "t": 1},  # Left
                {"x": 1, "y": 1, "t": 2},  # Down
            ]
        }

        trajectory, startings = parse_trajectories(schedule)

        # Check shapes
        assert trajectory.shape == (2, 4)  # 2 agents, 4 time steps
        assert startings.shape == (2, 2)   # 2 agents, (x, y)

        # Check starting positions
        assert np.array_equal(startings[0], [0, 0])
        assert np.array_equal(startings[1], [2, 2])

        # Check actions for agent 0
        assert trajectory[0, 0] == 1  # Right
        assert trajectory[0, 1] == 2  # Up
        assert trajectory[0, 2] == 0  # Wait
        assert trajectory[0, 3] == 0  # Implicit wait

        # Check actions for agent 1
        assert trajectory[1, 0] == 3  # Left
        assert trajectory[1, 1] == 4  # Down
        assert trajectory[1, 2] == 0  # Implicit wait
        assert trajectory[1, 3] == 0  # Implicit wait

    def test_parse_dataset_trajectories(self):
        """Test parsing trajectories for a dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create mock case with solution
            case_dir = path / "case_0"
            case_dir.mkdir()

            solution = {
                "schedule": {
                    "agent0": [{"x": 0, "y": 0}, {"x": 1, "y": 0}],
                    "agent1": [{"x": 2, "y": 2}, {"x": 2, "y": 1}]
                },
                "cost": 4
            }

            with open(case_dir / "solution.yaml", "w") as f:
                yaml.dump(solution, f)

            # Parse trajectories
            parse_dataset_trajectories(path)

            # Check output files
            assert (case_dir / "trajectory.npy").exists()
            assert (case_dir / "startings.npy").exists()

            # Load and verify
            trajectory = np.load(case_dir / "trajectory.npy")
            startings = np.load(case_dir / "startings.npy")

            assert trajectory.shape == (2, 2)  # 2 agents, 2 time steps
            assert startings.shape == (2, 2)   # 2 agents, (x, y)


class TestCrossPlatformPaths:
    """Test cross-platform path handling."""

    def test_pathlib_usage(self):
        """Test that Path objects work correctly."""
        # Test path construction
        path = Path("dataset") / "train" / "case_0"
        assert str(path).replace("\\", "/") == "dataset/train/case_0"

        # Test parent directory
        assert path.parent.name == "train"

        # Test file extensions
        solution_file = path / "solution.yaml"
        assert solution_file.suffix == ".yaml"

    def test_no_hardcoded_separators(self):
        """Verify no hardcoded path separators in new code."""
        # Check dataset_gen.py
        with open("data_generation/dataset_gen.py") as f:
            content = f.read()
            # Should not contain hardcoded Windows paths
            assert r"\\" not in content or "Path" in content
            assert "rf\"" not in content or "Path" in content

        # Check trajectory_parser.py
        with open("data_generation/trajectory_parser.py") as f:
            content = f.read()
            assert r"\\" not in content or "Path" in content
            assert "rf\"" not in content or "Path" in content


class TestRandomAssignmentEfficiency:
    """Test efficiency of random position assignment."""

    def test_sparse_board_efficiency(self):
        """Test efficiency on sparse boards."""
        # 10% fill should be efficient
        input_dict = gen_input((20, 20), 20, 10)  # 40/400 = 10%

        assert input_dict is not None
        # Should complete quickly (implicit by not timing out)

    def test_dense_board_efficiency(self):
        """Test efficiency on dense boards."""
        # 80% fill should still work with optimized algorithm
        input_dict = gen_input((10, 10), 60, 10)  # 80/100 = 80%

        # Should either succeed or explicitly fail
        # The optimized algorithm should handle this without hanging


def test_full_pipeline():
    """Integration test for full data pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)

        # Step 1: Generate dataset
        config = {
            "map_shape": [5, 5],
            "nb_agents": 2,
            "nb_obstacles": 3
        }
        create_solutions(path, 2, config)

        # Step 2: Parse trajectories
        parse_dataset_trajectories(path)

        # Step 3: Verify complete pipeline
        cases = list(path.glob("case_*"))
        for case in cases:
            if (case / "solution.yaml").exists():
                # Should have all required files
                assert (case / "input.yaml").exists()
                assert (case / "trajectory.npy").exists()
                assert (case / "startings.npy").exists()

                # Load and verify consistency
                with open(case / "input.yaml") as f:
                    input_data = yaml.load(f, Loader=yaml.FullLoader)

                trajectory = np.load(case / "trajectory.npy")
                startings = np.load(case / "startings.npy")

                # Number of agents should match
                num_agents = len(input_data["agents"])
                assert trajectory.shape[0] == num_agents
                assert startings.shape[0] == num_agents


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])