import torch
import torch.nn as nn
import torch.nn.functional as F
from utils_weights import weights_init
import numpy as np
from copy import copy


class Network(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # json file
        self.S = None
        self.num_agents = self.config["num_agents"]
        self.map_shape = self.config["map_shape"]  # FOV
        self.num_actions = 5

        # dim_encoder_mlp = 1
        dim_encoder_mlp = self.config["encoder_layers"]
        # self.compress_Features_dim = [128]  # Check
        self.compress_Features_dim = self.config["encoder_dims"]  # Check

        # dim_action_mlp = 1
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
        conv_features_dim = (
            channels[-1] * self.conv_dim_W[-1] * self.conv_dim_H[-1]
        )  # this is the dimension of the features after the convolutional layers
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

        ############################################################
        # MLP Action
        ############################################################

        action_features = [self.compress_Features_dim[-1]] + action_features

        mlp_action = []
        for l in range(dim_action_mlp):
            if l < dim_action_mlp - 1:
                mlp_action.append(nn.Linear(action_features[l], action_features[l + 1]))
                mlp_action.append(nn.ReLU(inplace=True))
            else:
                mlp_action.append(nn.Linear(action_features[l], action_features[l + 1]))

        self.actionMLP = nn.Sequential(*mlp_action)
        self.apply(weights_init)

    def forward(self, states):
        """
        states.shape = (batch x agent  x channels x dimX x dimY)
        """
        batch_size = states.shape[0]
        # This vector is only needed for the GNN
        # feature_vector = torch.zeros(batch_size, self.compress_features_dim[-1], self.num_agents).to(self.config["device"])
        action_logits = torch.zeros(batch_size, self.num_agents, self.num_actions).to(
            self.config["device"]
        )
        for id_agent in range(self.num_agents):
            agent_state = states[:, id_agent, :, :, :]
            features = self.convLayers(agent_state)
            features_flatten = features.view(features.size(0), -1)  # B x T*C*W*H
            encoded_feats = self.compressMLP(features_flatten)
            # feature_vector[:, :, id_agent] # B x F x N
            encoded_feats_flat = encoded_feats.view(encoded_feats.size(0), -1)
            action_agent = self.actionMLP(encoded_feats_flat)  # 1 x 5
            action_logits[:, id_agent, :] = action_agent

        return action_logits
