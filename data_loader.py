import os
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader

class GNNDataLoader:
    def __init__(self, config):
        self.config = config
    
        train_set = CreateDataset(self.config, "train")
        valid_set = CreateDataset(self.config, "valid")
        # test_trainingSet = CreateDataset(self.config, "test_trainingSet")
        # validStep_set = CreateDataset(self.config, "validStep")

        self.train_loader = DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True,
                                        num_workers=self.config["num_workers"], pin_memory=True)
        self.valid_loader = DataLoader(valid_set, batch_size=self.config["batch_size"], shuffle=True,
                                        num_workers=self.config["num_workers"], pin_memory=True)
        # self.train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True,
        #                                num_workers=self.config.data_loader_workers,
        #                                pin_memory=self.config.pin_memory)
        # self.validStep_loader = DataLoader(validStep_set, batch_size=self.config.batch_size, shuffle=True,
        #                                num_workers=self.config.data_loader_workers,
        #                                pin_memory=self.config.pin_memory)

        # self.test_trainingSet_loader = DataLoader(test_trainingSet, batch_size=self.config.valid_batch_size, shuffle=True,
        #                                num_workers=self.config.data_loader_workers,
        #                                pin_memory=self.config.pin_memory)


class CreateDataset(data.Dataset):
   
    def __init__(self,config, mode):
        """
        Args:
            dir_path (string): Path to the directory with the cases.
            A case dir contains the states and trajectories of the agents
        """
        self.config = config[mode]
        self.dir_path = self.config["root_dir"]
        if mode == "valid":
            self.dir_path = os.path.join(self.dir_path, "val")
        elif mode == "train":
            self.dir_path = os.path.join(self.dir_path, "train")
        self.cases = os.listdir(self.dir_path)
        self.states         = np.zeros((len(self.cases),self.config["max_time"] ,self.config["nb_agents"], 2, 5, 5)) # case x time x agent x channels x dimX x dimy
        self.trajectories   = np.zeros((len(self.cases),self.config["max_time"] ,self.config["nb_agents"])) # case x time x agent

        for i, case in enumerate(self.cases):
            states = np.load(os.path.join(self.dir_path, case, "states.npy"))
            states = states[1:,:,:,:,:]
            tray = np.load(os.path.join(self.dir_path, case, "trajectory.npy"))
            assert states.shape[0] == tray.shape[1], f"(before transform) Missmatch between states and trajectories: {states.shape[0]} != {tray.shape[1]}"
            self.states[i,:,:,:,:,:] = states
            self.trajectories[i, :, :] = tray.reshape((self.config["max_time"], self.config["nb_agents"]))
        
        self.states = self.states.reshape((-1, self.config["nb_agents"], 2, 5, 5))
        self.trajectories = self.trajectories.reshape((-1,self.config["nb_agents"]))
        assert self.states.shape[0] == self.trajectories.shape[0], f"(after transform) Missmatch between states and trajectories: {states.shape[0]} != {tray.shape[0]}"


    def __len__(self):
        return self.states.shape[0]
    
    def __getitem__(self, index):
        """
        Returns the whole case 
        state : (time, agents, channels, dimX, dimY),
        trayec: (time, agents)
        """
        states = torch.from_numpy(self.states[index]).float()
        trayec = torch.from_numpy(self.trajectories[index]).float()
        return states, trayec
    
    # def device(self, device):
    #     self.to(device)

if __name__ == "__main__":
    config = {
        "train":{
                "root_dir": r"dataset\2_0_8", 
                "mode": "train",
                "max_time": 13,
                "nb_agents": 2,
              },
        "valid":{
                "root_dir": r"dataset\2_0_8", 
                "mode": "train",
                "max_time": 15,
                "nb_agents": 2,
              }
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
