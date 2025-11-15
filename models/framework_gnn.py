from typing import Dict, Any
import torch
import torch.nn as nn
from networks.utils_weights import weights_init
from networks.gnn import GCNLayer
from copy import copy


class Network(nn.Module):
    """Graph Neural Network for decentralized multi-agent path planning.

    This model implements a 3-stage architecture:
    1. CNN encoder to process each agent's local field-of-view
    2. Graph Convolutional Network to enable inter-agent communication
    3. MLP policy head to output action probabilities

    Args:
        config: Dictionary containing model configuration parameters
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.S = None
        self.num_agents = self.config["num_agents"]
        self.map_shape = self.config["map_shape"]  # FOV
        self.num_actions = 5

        dim_encoder_mlp = self.config["encoder_layers"]
        self.compress_Features_dim = self.config["encoder_dims"]  # Check

        self.graph_filter = self.config["graph_filters"]
        self.node_dim = self.config["node_dims"]

        dim_action_mlp = self.config["action_layers"]

        action_features = [self.num_actions]

        ############################################################
        # CNN
        ############################################################

        self.conv_dim_W = []
        self.conv_dim_H = []
        self.conv_dim_W.append(self.map_shape[0])
        self.conv_dim_H.append(self.map_shape[1])

        # channels = [2] + [32, 32, 64, 64, 128]
        channels = [2] + self.config["channels"]
        num_conv = len(channels) - 1
        strides = [1, 1, 1, 1, 1]
        padding_size = [1] * num_conv
        filter_taps = [3] * num_conv

        conv_layers = []
        H_tmp = copy(self.map_shape[0])
        W_tmp = copy(self.map_shape[1])
        for l in range(num_conv):

            conv_layers.append(
                nn.Conv2d(
                    in_channels=channels[l],
                    out_channels=channels[l + 1],
                    kernel_size=filter_taps[l],
                    stride=strides[l],
                    padding=padding_size[l],
                    bias=True,
                )
            )
            conv_layers.append(nn.BatchNorm2d(num_features=channels[l + 1]))
            conv_layers.append(nn.ReLU(inplace=True))

            W_tmp = int((W_tmp - filter_taps[l] + 2 * padding_size[l]) / strides[l]) + 1
            H_tmp = int((H_tmp - filter_taps[l] + 2 * padding_size[l]) / strides[l]) + 1

            self.conv_dim_W.append(W_tmp)
            self.conv_dim_H.append(H_tmp)

        self.convLayers = nn.Sequential(*conv_layers)
        # this is the dimension of the features after the convolutional layers
        ############################################################
        # MLP Encoder
        ############################################################

        # self.compress_Features_dim = [conv_features_dim] + self.compress_Features_dim
        self.compress_Features_dim = (
            self.config["last_convs"] + self.compress_Features_dim
        )

        mlp_encoder = []
        for l in range(dim_encoder_mlp):
            mlp_encoder.append(
                nn.Linear(
                    self.compress_Features_dim[l], self.compress_Features_dim[l + 1]
                )
            )
            mlp_encoder.append(nn.ReLU(inplace=True))

        self.compressMLP = nn.Sequential(*mlp_encoder)
        self.shared_feats = self.compress_Features_dim[-1]

        ############################################################
        # GNN
        ############################################################

        self.nb_filter = len(self.graph_filter)  # Number of graph filtering layers
        self.features = [self.compress_Features_dim[-1]] + self.node_dim  # Features
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
        self.graph_filter  # nFilterTaps # Filter taps
        gcn = []
        for l in range(self.nb_filter):
            gcn.append(
                GCNLayer(
                    self.num_agents,
                    self.features[l],
                    self.features[l + 1],
                    self.graph_filter[l],
                    activation=None,
                )
            )
            gcn.append(nn.ReLU(inplace=True))
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gcn)  # Graph Filtering Layers

        ############################################################
        # MLP Action
        ############################################################

        # action_features = [self.compress_Features_dim[-1]] + action_features
        action_features = [self.features[-1]] + action_features

        mlp_action = []
        for l in range(dim_action_mlp):
            if l < dim_action_mlp - 1:
                mlp_action.append(nn.Linear(action_features[l], action_features[l + 1]))
                mlp_action.append(nn.ReLU(inplace=True))
            else:
                mlp_action.append(nn.Linear(action_features[l], action_features[l + 1]))

        self.actionMLP = nn.Sequential(*mlp_action)
        self.apply(weights_init)

    def forward(self, states: torch.Tensor, gso: torch.Tensor) -> torch.Tensor:
        """Forward pass through the GNN model.

        Args:
            states: Agent observations [batch_size, num_agents, channels, height, width]
                    - channel 0: obstacles and other agents
                    - channel 1: goal locations
            gso: Graph shift operator (adjacency matrix) [batch_size, num_agents, num_agents]

        Returns:
            Action logits for each agent [batch_size, num_agents, num_actions]
        """
        batch_size = states.shape[0]
        # This vector is only needed for the GNN
        feature_vector = torch.zeros(
            batch_size, self.compress_Features_dim[-1], self.num_agents
        ).to(self.config["device"])
        for id_agent in range(self.num_agents):
            agent_state = states[:, id_agent, :, :, :]
            features = self.convLayers(agent_state)
            features_flatten = features.view(features.size(0), -1)  # B x T*C*W*H
            encoded_feats = self.compressMLP(features_flatten)
            encoded_feats_flat = encoded_feats.view(encoded_feats.size(0), -1)
            feature_vector[:, :, id_agent] = encoded_feats_flat  # B x F x N

        for layer in range(self.nb_filter):
            self.GFL[layer * 2].addGSO(gso)

        features_shared = self.GFL(feature_vector)  # B x F x N

        action_logits = torch.zeros(batch_size, self.num_agents, self.num_actions).to(
            self.config["device"]
        )
        for id_agent in range(self.num_agents):
            agent_shared = features_shared[:, :, id_agent]
            action_agent = self.actionMLP(agent_shared)  # 1 x 5
            action_logits[:, id_agent, :] = action_agent

        return action_logits


if __name__ == "__main__":
    config = {
        "device": "cpu",
        "map_shape": [5, 5],
        "num_agents": 4,
        "num_actions": 5,
        "last_convs": [400],
        "channels": [2, 16, 16],
        "node_dims": [128],
        "graph_filters": [2],
        "encoder_dims": [64],
        "dim_action_mlp": 1,
        "encoder_layers": 1,
        "action_layers": 1,
        "compress_Features_dim": [64],
    }
    S = torch.eye(4).unsqueeze(0)
    S[:, 0, 0] = 10
    print(S)

    gf = Network(
        config=config,
    )

    print(gf)
    # states.shape = (batch x agent  x channels x dimX x dimY)
    states = torch.ones(size=(1, 4, 2, 5, 5))

    # x is of shape: batchSize x dimInFeatures x numberNodesIn
    x_t = gf(states, S)
    print(x_t)
