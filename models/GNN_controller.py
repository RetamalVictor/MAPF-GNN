import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_weights import weights_init
import numpy as np

class GNNControl(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config #json file
        self.S = None
        self.numAgents = self.config.num_agents
        self.map_shape = self.config.map_shape
        
        convW = [self.map_shape[0]]
        convH = [self.map_shape[1]]

        numAction = 5
        numChannel = [3] + [32, 32, 64, 64, 128]
        numStride = [1, 1, 1, 1, 1]

        dimCompressMLP = 1
        numCompressFeatures = [2 ** 7]

        nMaxPoolFilterTaps = 2
        numMaxPoolStride = 2

        ############################################################
        # CNN
        ############################################################

        convl = []
        numConv = len(numChannel) - 1
        nFilterTaps = [3] * numConv
        nPaddingSzie = [1] * numConv
        for l in range(numConv):
            convl.append(nn.Conv2d(in_channels=numChannel[l], out_channels=numChannel[l + 1],
                                    kernel_size=nFilterTaps[l], stride=numStride[l], padding=nPaddingSzie[l],
                                    bias=True))
            convl.append(nn.BatchNorm2d(num_features=numChannel[l + 1]))
            convl.append(nn.ReLU(inplace=True))

            W_tmp = int((convW[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            H_tmp = int((convH[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
            # Adding maxpooling
            if l % 2 == 0:
                convl.append(nn.MaxPool2d(kernel_size=2))
                W_tmp = int((W_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                H_tmp = int((H_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                # http://cs231n.github.io/convolutional-networks/
            convW.append(W_tmp)
            convH.append(H_tmp)

        self.ConvLayers = nn.Sequential(*convl)

        numFeatureMap = numChannel[-1] * convW[-1] * convH[-1]

        ############################################################
        # MLP
        ############################################################

    def forward(self, X):
        