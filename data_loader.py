import os
import numpy as np

import torch
from torch.utils import data
from torch.utils.data import DataLoader

class GNNDataLoader:
    def __init__(self, config):
        self.config = config
    
        train_set = CreateDataset(self.config, "train")

        self.train_loader = DataLoader(train_set, batch_size=self.config["batch_size"], shuffle=True,
                                        num_workers=self.config["num_workers"], pin_memory=True)



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
        self.states         = np.zeros((len(self.cases),self.config["min_time"] ,self.config["nb_agents"], 2, 5, 5)) # case x time x agent x channels x dimX x dimy
        self.trajectories   = np.zeros((len(self.cases),self.config["min_time"] ,self.config["nb_agents"])) # case x time x agent
        self.gsos           = np.zeros((len(self.cases),self.config["min_time"] , self.config["nb_agents"], self.config["nb_agents"])) # case x time x agent x nodes x nodes
        self.count = 0
        
        for i, case in enumerate(self.cases):
            if os.path.exists(os.path.join(self.dir_path, case, "states.npy")):
                state = np.load(os.path.join(self.dir_path, case, "states.npy"))
                state = state[1:self.config["min_time"]+1,:,:,:,:]
                tray = np.load(os.path.join(self.dir_path, case, "trajectory_record.npy"))
                tray = tray[:,:self.config["min_time"]]
                gso = np.load(os.path.join(self.dir_path, case, "gso.npy"))
                gso = gso[:self.config["min_time"],0,:,:] # select the first agent since all agents have the same gso
                gso = gso + np.eye(self.config["nb_agents"]) # add self loop
                if state.shape[0] < self.config["min_time"] or tray.shape[1] < self.config["min_time"]:
                    continue
                if state.shape[0] > self.config["max_time_dl"] or tray.shape[1] > self.config["max_time_dl"]:
                    continue
                assert state.shape[0] == tray.shape[1], f"(before transform) Missmatch between states and trajectories: {state.shape[0]} != {tray.shape[1]}"
                self.states[i,:,:,:,:,:] = state
                # self.trajectories[i, :, :] = tray.reshape((self.config["max_time"], self.config["nb_agents"]))
                self.trajectories[i, :, :] = tray.T
                self.gsos[i,:,:,:] = gso
                self.count += 1

        self.states = self.states[:self.count,:,:,:,:,:]
        self.trajectories = self.trajectories[:self.count,:,:]
        self.gsos = self.gsos[:self.count,:,:,:]
        self.states = self.states.reshape((-1, self.config["nb_agents"], 2, 5, 5))
        self.trajectories = self.trajectories.reshape((-1,self.config["nb_agents"]))
        self.gsos = self.gsos.reshape((-1, self.config["nb_agents"], self.config["nb_agents"]))
        assert self.states.shape[0] == self.trajectories.shape[0], f"(after transform) Missmatch between states and trajectories: {state.shape[0]} != {tray.shape[0]}"
        print(f"Zeros: {self.statistics()}")
        print(f"Loaded {self.count} cases")
        
    def statistics(self):
        zeros = np.count_nonzero(self.trajectories == 0)
        return zeros / (self.trajectories.shape[0] * self.trajectories.shape[1])

    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        """
        Returns the whole case 
        state : (time, agents, channels, dimX, dimY),
        trayec: (time, agents)
        gsos: (time, agents, nodes, nodes)
        """
        states = torch.from_numpy(self.states[index]).float()
        trayec = torch.from_numpy(self.trajectories[index]).float()
        gsos   = torch.from_numpy(self.gsos[index]).float()
        return states, trayec, gsos
    
    # def device(self, device):
    #     self.to(device)

if __name__ == "__main__":
    config = {
        "train":{
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
