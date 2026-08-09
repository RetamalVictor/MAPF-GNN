import os
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader


class GNNDataLoader:
    def __init__(self, config):
        self.config = config

        train_set = CreateDataset(self.config, "train")

        self.train_loader = DataLoader(
            train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["num_workers"],
            pin_memory=True,
        )


class CreateDataset(data.Dataset):
    def __init__(self, config, mode):
        """
        Args:
            dir_path (string): Path to the directory with the cases.
            A case dir contains the states and trajectories of the agents
        """
        self.config = config[mode]

        # root_dir may be a single path or several. The online expert writes its
        # relabelled cases to their own directory, and training on both means
        # listing both rather than copying one into the other. case_caps limits
        # how many cases each root contributes, which is how the share of
        # online-expert data in the set is held down.
        roots = self.config["root_dir"]
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        caps = self.config.get("case_caps") or [None] * len(roots)

        subdir = {"valid": "val", "train": "train"}.get(mode)
        self.dir_paths = [
            os.path.join(root, subdir) if subdir else str(root) for root in roots
        ]
        self.dir_path = self.dir_paths[0]

        self.cases = []
        for dir_path, cap in zip(self.dir_paths, caps):
            names = sorted(os.listdir(dir_path))
            if cap is not None:
                names = names[:cap]
            self.cases.extend(os.path.join(dir_path, name) for name in names)

        min_time = self.config["min_time"]
        # Optional cap on how much of each trajectory to keep. Cases run for ~30
        # steps on average, so a small cap here means the policy only ever sees
        # the opening moves and never learns to approach a goal.
        max_time = self.config.get("max_time_dl")

        states, trajectories, gsos = [], [], []
        self.count = 0

        for case_dir in self.cases:
            if not os.path.exists(os.path.join(case_dir, "states.npy")):
                continue

            state = np.load(os.path.join(case_dir, "states.npy"))
            tray = np.load(os.path.join(case_dir, "trajectory_record.npy"))
            gso = np.load(os.path.join(case_dir, "gso.npy"))
            # All agents record the same graph; keep one copy.
            gso = gso[:, 0, :, :]

            # states[t] is the observation the expert acted on at step t, so it
            # pairs with tray[:, t]. Any offset here trains the policy to predict
            # the action that produced the current state instead of the next one.
            steps = min(state.shape[0], tray.shape[1], gso.shape[0])
            if steps < min_time:
                continue
            if max_time:
                steps = min(steps, max_time)

            states.append(state[:steps])
            trajectories.append(tray[:, :steps].T)
            # No self-loop added here: the GCN layer adds its own, and evaluation
            # feeds the raw adjacency straight from the environment.
            gsos.append(gso[:steps])
            self.count += 1

        if not self.count:
            raise RuntimeError(f"No recorded cases found in {self.dir_path}")

        self.states = np.concatenate(states, axis=0)
        self.trajectories = np.concatenate(trajectories, axis=0)
        self.gsos = np.concatenate(gsos, axis=0)

        assert (
            self.states.shape[0] == self.trajectories.shape[0] == self.gsos.shape[0]
        ), (
            "Mismatch between states, trajectories and graphs: "
            f"{self.states.shape[0]} / {self.trajectories.shape[0]} / {self.gsos.shape[0]}"
        )
        print(f"Zeros: {self.statistics()}")
        print(f"Loaded {self.count} cases ({self.states.shape[0]} timesteps)")

    def statistics(self):
        zeros = np.count_nonzero(self.trajectories == 0)
        return zeros / (self.trajectories.shape[0] * self.trajectories.shape[1])

    def __len__(self):
        # Each case contributes min_time timesteps, and the arrays above are
        # flattened to one sample per timestep. Returning self.count here would
        # hide every timestep after the first few from the sampler.
        return self.states.shape[0]

    def __getitem__(self, index):
        """
        Returns the whole case
        state : (time, agents, channels, dimX, dimY),
        trayec: (time, agents)
        gsos: (time, agents, nodes, nodes)
        """
        states = torch.from_numpy(self.states[index]).float()
        trayec = torch.from_numpy(self.trajectories[index]).float()
        gsos = torch.from_numpy(self.gsos[index]).float()
        return states, trayec, gsos

    # def device(self, device):
    #     self.to(device)


if __name__ == "__main__":
    config = {
        "train": {
            "root_dir": r"dataset\2_0_6v2",
            "mode": "train",
            "max_time": 13,
            "nb_agents": 2,
            "min_time": 13,
        },
    }

    data_loader = GNNDataLoader(config)
    print(data_loader.train_loader)
    train_features, train_labels = next(iter(data_loader.train_loader))
    print("Train:")
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    valid_features, valid_labels = next(iter(data_loader.valid_loader))
    print("Valid:")
    print(f"Feature batch shape: {valid_features.size()}")
    print(f"Labels batch shape: {valid_labels.size()}")
