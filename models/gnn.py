import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
from scipy.special import softmax
import math

def glorot_init(nin, nout):
  sd = np.sqrt(6.0 / (nin + nout))
  return np.random.uniform(-sd, sd, size=(nin,nout))


class GCNLayer(nn.Module):

    def __init__(self, n_nodes, in_features, out_features, filter_number, bias=False, activation=None, name='GCN_Layer'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.filter_number = filter_number
        self.W = nn.parameter.Parameter(torch.Tensor(self.in_features,self.filter_number, self.out_features))
        if bias:
            self.b = nn.parameter.Parameter(torch.Tensor(self.out_features))
        else: 
            self.register_parameter('bias', None)
            self.b = None
        self.activation = activation
        self.name = name
        self.init_params()

    def init_params(self):
        stdv = 1. / math.sqrt(self.in_features * self.filter_number)
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.in_features, self.out_features) + "filter_taps=%d, " % (
                        self.filter_number) + \
                        "bias=%s, " % (self.b is not None)
        return reprString

    def addGSO(self, GSO):
        self.adj_matrix = GSO

    def forward(self, node_feats):
        """
        node_feats: [batch_size, n_nodes, in_features]
        adj_matrix: [batch_size, n_nodes, n_nodes]
        """
        self.n_nodes = node_feats.shape[2]
        batch_size = node_feats.shape[0]
        self.adj_matrix += torch.eye(self.n_nodes)
        # Generating an empty D
        D_mod = torch.zeros_like(self.adj_matrix)

        # Filling it with the total number of neigh (plus self connections)
        k = D_mod.size(1)
        for i in range(batch_size):
          D_mod[i].as_strided([k], [k + 1]).copy_(self.adj_matrix[i].sum(axis=1).flatten())
        
        # torch.fill_diagonal(D_mod, torch.tensor(adj_matrix.sum(axis=1)).flatten())
        D_mod_invroot = torch.pow(D_mod, -0.5)
        D_mod_invroot[D_mod_invroot == torch.inf] = 0
        adj_matrix = D_mod_invroot @ self.adj_matrix @ D_mod_invroot

        # K-hops
        node_feats = node_feats.reshape([batch_size, self.in_features, self.n_nodes])
        z = node_feats.reshape([batch_size,1,self.in_features,self.n_nodes]) 
        for k in range(1, self.filter_number):
          node_feats = node_feats @ adj_matrix
          xS = node_feats.reshape([batch_size, 1, self.in_features, self.n_nodes])
          z = torch.cat((z, xS), dim=2)
        
        z = z.permute(0, 3, 1, 2).reshape([batch_size, self.n_nodes, self.filter_number*self.in_features])
        W = self.W.reshape([self.out_features, self.filter_number*self.in_features]).permute(1,0)
        node_feats = z @ W
        node_feats = node_feats.permute(0,2,1)


        if self.b is not None:
            node_feats += node_feats + self.b
        # node_feats = node_feats / n_neig
        if self.activation is not None:
            node_feats = self.activation(node_feats)

        return node_feats


# class GCNLayer():
#   def __init__(self,n_nodes, in_features, out_features, activation=None, name=''):
#     self.in_features = in_features
#     self.out_features = out_features
#     self.n_nodes = n_nodes
#     self.W = glorot_init(self.in_features, self.out_features)
#     self.activation = activation
#     self.name = name
  
#   def __repr__(self):
#     return f"GCN: W{'_' + self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"

#   def forward(self, adj_matrix, node_feats, W=None, b=None):

#     if W is None:
#       W = self.W
#     adj_matrix += np.eye(self.n_nodes)
#     # n_neig = adj_matrix.sum(axis=-1, keepdims=True)

#     # Generating an empty D
#     D_mod = np.zeros_like(adj_matrix)
#     # Filling it with the total number of neigh (plus self connections)
#     np.fill_diagonal(D_mod, np.asarray(adj_matrix.sum(axis=1)).flatten())
#     D_mod_invroot = np.linalg.inv(sqrtm(D_mod))

#     adj_matrix = D_mod_invroot @ adj_matrix @ D_mod_invroot
#     node_feats = adj_matrix @ node_feats @ W
#     if b is not None:
#       node_feats += node_feats + b
#     # node_feats = node_feats / n_neig
#     if self.activation is not None:
#       node_feats = self.activation(node_feats)
#     return node_feats


if __name__ == "__main__":
    node_feats = np.arange(16, dtype=np.float32).reshape((2,4, 2))
    adj_matrix = np.array([[[1, 1, 0, 0],
                                [1, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]],
                          [[1, 1, 0, 0],
                                [1, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]]], dtype=np.float32)
    n_nodes = adj_matrix.shape[2]
    node_feats = torch.from_numpy(node_feats)
    adj_matrix = torch.from_numpy(adj_matrix)
    in_features = node_feats.shape[2]
    out_features = 2
    print(f"Node features shape: {node_feats.shape}")
    print(f"Adjacency matrix shape: {adj_matrix.shape}")

    # W = np.random.uniform(size=(node_feats.shape[-1], out_features))
    # W = np.array([[1., 0.], [0., 1.]])
    activation = nn.ReLU()
    K=2
    gcn = GCNLayer(n_nodes, in_features, out_features, filter_number=K, activation=activation, name='test', bias=True)
    print(gcn)
    gcn.addGSO(adj_matrix)
    node_feats = gcn(node_feats)
    print(node_feats)


    # print("Node features:\n", node_feats)
    # print("\nAdjacency matrix:\n", adj_matrix)
    # D_mod = np.zeros_like(adj_matrix)
    # print(f"D_mod shape: {D_mod.shape}")
    # # Filling it with the total number of neigh (plus self connections)
    # np.fill_diagonal(D_mod[0], np.asarray(adj_matrix[0].sum(axis=1)).flatten())
    # D_mod_invroot = np.linalg.inv(sqrtm(D_mod[0]))
    # adj_matrix = D_mod_invroot @ adj_matrix @ D_mod_invroot
    # print("\nAdjacency matrix:\n", adj_matrix)
    # print("\n")
    # neigh = adj_matrix.sum(axis=2, keepdims=True)
    # print(neigh)
    # neigh = adj_matrix.sum(axis=-1, keepdims=True)
    # print(neigh)
    # Z_temp = adj_matrix @ node_feats

    # print("\n",Z_temp@W)