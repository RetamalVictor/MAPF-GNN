from dataset_gen import create_solutions
from trajectory_parser import parse_dataset_trajectories
from record import record_env


if __name__ == "__main__":
    cases = 5  # Small test run
    config = {
        "num_agents": 3,  # Fewer agents for quick test
        "map_shape": [16, 16],  # Smaller grid for faster solving
        "nb_agents": 3,
        "nb_obstacles": 4,
        "sensor_range": 4,
        "board_size": [16, 16],
        "max_time": 20,
        "min_time": 8,  # min time the tray should go from start to goal
        "path": "dataset/test_3_4_16/train",  # Test dataset path
    }

    for path in [config["path"]]:
        create_solutions(path, cases, config)
        parse_dataset_trajectories(path)
        record_env(path, config)
