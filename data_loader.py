import os
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader

class DecentralPlannerDataLoader:
    def __init__(self, config):
        self.config = config
        if config["mode"] == "train":

            train_set = CreateDataset(self.config)
            # test_trainingSet = CreateDataset(self.config, "test_trainingSet")
            # validStep_set = CreateDataset(self.config, "validStep")
            # valid_set = CreateDataset(self.config, "valid")

            self.train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                                           num_workers=3)
            # self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
            #                                num_workers=self.config.data_loader_workers,
            #                                pin_memory=self.config.pin_memory)
            # self.validStep_loader = DataLoader(validStep_set, batch_size=self.config.batch_size, shuffle=True,
            #                                num_workers=self.config.data_loader_workers,
            #                                pin_memory=self.config.pin_memory)

            # self.test_trainingSet_loader = DataLoader(test_trainingSet, batch_size=self.config.valid_batch_size, shuffle=True,
            #                                num_workers=self.config.data_loader_workers,
            #                                pin_memory=self.config.pin_memory)
            # self.valid_loader = DataLoader(valid_set, batch_size=self.config.valid_batch_size, shuffle=True,
            #                                num_workers=self.config.data_loader_workers,
            #                                pin_memory=self.config.pin_memory)


class CreateDataset(data.Dataset):
   
    def __init__(self,config):
        """
        Args:
            dir_path (string): Path to the directory with the cases.
            A case dir contains the states and trajectories of the agents
        """
        self.config = config
        self.dir_path = config["root_dir"]
        self.cases = os.listdir(self.dir_path)
        self.states         = np.zeros((len(self.cases),self.config["max_time"] ,self.config["nb_agents"], 2, 5, 5)) # case x time x agent x channels x dimX x dimy
        self.trajectories   = np.zeros((len(self.cases),self.config["max_time"] ,self.config["nb_agents"])) # case x time x agent

        for i, case in enumerate(self.cases):
            states = np.load(os.path.join(self.dir_path, case, "states.npy"))
            states = states[1:,:,:,:,:]
            tray = np.load(os.path.join(self.dir_path, case, "trajectory.npy"))
            assert states.shape[0] == tray.shape[1], f"(before transform) Missmatch between states and trajectories: {states.shape[0]} != {tray.shape[1]}"
            self.states[i,:,:,:,:,:] = states
            self.trajectories[i, :, :] = tray.reshape((8, 2))
        
        self.states = self.states.reshape((-1, 2, 5, 5))
        self.trajectories = self.trajectories.reshape((-1,1))
        assert self.states.shape[0] == self.trajectories.shape[0], f"(after transform) Missmatch between states and trajectories: {states.shape[0]} != {tray.shape[0]}"


    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, index):
        """
        Returns the whole case 
        state : (time, agents, channels, dimX, dimY),
        trayec: (time, agents)
        """
        return self.states[index], self.trajectories[index]

if __name__ == "__main__":
    config = {
            "root_dir": r"dataset\2_0_5", 
            "mode": "train",
            "max_time": 8,
            "nb_agents": 2,
              }
    train_data = CreateDataset(config)
    print(len(train_data))
    print(train_data[0][0].shape)
    print(train_data[0][1].shape)
    data_loader = DecentralPlannerDataLoader(config)
    print(data_loader.train_loader)
    train_features, train_labels = next(iter(data_loader.train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
