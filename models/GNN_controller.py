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
        self.numAgents = self.config["num_agents"]
        self.map_shape = self.config["map_shape"]
        
        convW = [self.map_shape[0]]
        convH = [self.map_shape[1]]

        numAction = 5
        # numChannel = [3] + [32, 32, 64, 64, 128]
        numChannel = [2] + [32, 32, 64, 64, 128]
        numStride = [1, 1, 1, 1, 1]

        dimCompressMLP = 1
        numCompressFeatures = [2 ** 7]

        nMaxPoolFilterTaps = 2
        numMaxPoolStride = 2
        # --- actionMLP
        dimActionMLP = 1
        numActionFeatures = [numAction]

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
        numCompressFeatures = [numFeatureMap] + numCompressFeatures

        compressmlp = []
        for l in range(dimCompressMLP):
            compressmlp.append(
                nn.Linear(in_features=numCompressFeatures[l], out_features=numCompressFeatures[l + 1], bias=True))
            compressmlp.append(nn.ReLU(inplace=True))

        self.compressMLP = nn.Sequential(*compressmlp)

        self.numFeatures2Share = numCompressFeatures[-1]
        #####################################################################
        # MLP --- map to actions                         
        #####################################################################
        # numActionFeatures = [self.F[-1]] + numActionFeatures
        numActionFeatures = [numCompressFeatures[-1]] + numActionFeatures
        actionsfc = []
        for l in range(dimActionMLP):
            if l < (dimActionMLP - 1):
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))
                actionsfc.append(nn.ReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))

        self.actionsMLP = nn.Sequential(*actionsfc)
        self.apply(weights_init)
        
    # def forward(self, X):
    def forward(self, inputTensor):

        B = inputTensor.shape[0] # batch size

        # B x G x N
        extractFeatureMap = torch.zeros(B, self.numFeatures2Share, self.numAgents).to(self.config["device"])
        for id_agent in range(self.numAgents):
            input_currentAgent = inputTensor[:, id_agent]
            featureMap = self.ConvLayers(input_currentAgent)
            featureMapFlatten = featureMap.view(featureMap.size(0), -1)
            # extractFeatureMap[:, :, id_agent] = featureMapFlatten
            compressfeature = self.compressMLP(featureMapFlatten)
            extractFeatureMap[:, :, id_agent] = compressfeature # B x F x N

        # DCP
        # for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            # self.GFL[2 * l].addGSO(self.S) # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        # sharedFeature = self.GFL(extractFeatureMap)

        action_predict = []
        for id_agent in range(self.numAgents):
            # DCP_nonGCN
            # sharedFeature_currentAgent = extractFeatureMap[:, :, id_agent]
            # DCP
            # torch.index_select(sharedFeature_currentAgent, 3, id_agent)
            # sharedFeature_currentAgent = sharedFeature[:, :, id_agent]
            sharedFeature_currentAgent = extractFeatureMap[:, :, id_agent]
            # print("sharedFeature_currentAgent.requires_grad: {}\n".format(sharedFeature_currentAgent.requires_grad))
            # print("sharedFeature_currentAgent.grad_fn: {}\n".format(sharedFeature_currentAgent.grad_fn))

            sharedFeatureFlatten = sharedFeature_currentAgent.view(sharedFeature_currentAgent.size(0), -1)
            action_currentAgents = self.actionsMLP(sharedFeatureFlatten) # 1 x 5
            action_predict.append(action_currentAgents) # N x 5
        return action_predict

if __name__ == "__main__":
    config = {
        "device":"cpu",
        "num_agents":3,
        "map_shape":[10,10]
    }
    model = GNNControl(config)
    print(model)
    B=2
    C=2
    M=10
    N=3
    X = torch.randn(3, 2, 10, 10).unsqueeze(0)
    # X = torch.from_numpy(X).float().unsqueeze(0)
    print(X.shape)
    x_t = model.forward(X)
    print(x_t)
    fun_soft = nn.LogSoftmax(dim=-1)
    action = fun_soft(x_t[1])
    actionKey_predict = torch.max(action, 1)[1]
    print(actionKey_predict)