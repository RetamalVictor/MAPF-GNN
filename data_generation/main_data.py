from dataset_gen import create_solutions
from trayectory_parser import parse_traject
from record import record_env


if __name__ == "__main__":
    cases=1000
    config = {
        "device":"cpu",
        "num_agents":5,
        "map_shape":[12,12],
        "nb_agents": 5,
        "nb_obstacles": 0
        }

    for path in [fr"dataset\5_0_12v2\train"]:
        create_solutions(path, cases, config)
        parse_traject(path)
        record_env(path)