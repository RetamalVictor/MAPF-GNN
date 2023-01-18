import numpy as np
from scipy.linalg import sqrtm
from scipy.special import softmax

def glorot_init(nin, nout):
  sd = np.sqrt(6.0 / (nin + nout))
  return np.random.uniform(-sd, sd, size=(nin,nout))

class GCNLayer():
  def __init__(self,n_nodes, in_features, out_features, activation=None, name=''):
    self.in_features = in_features
    self.out_features = out_features
    self.n_nodes = n_nodes
    self.W = glorot_init(self.in_features, self.out_features)
    self.activation = activation
    self.name = name
  
  def __repr__(self):
    return f"GCN: W{'_' + self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"

  def forward(self, adj_matrix, node_feats, W=None):

    if W is None:
      W = self.W
    adj_matrix += np.eye(self.n_nodes)
    # n_neig = adj_matrix.sum(axis=-1, keepdims=True)

    # Generating an empty D
    D_mod = np.zeros_like(adj_matrix)
    # Filling it with the total number of neigh (plus self connections)
    np.fill_diagonal(D_mod, np.asarray(adj_matrix.sum(axis=1)).flatten())
    D_mod_invroot = np.linalg.inv(sqrtm(D_mod))

    adj_matrix = D_mod_invroot @ adj_matrix @ D_mod_invroot
    node_feats = adj_matrix @ node_feats @ W

    # node_feats = node_feats / n_neig
    if self.activation is not None:
      node_feats = self.activation(node_feats)
    return node_feats


if __name__ == "__main__":
    node_feats = np.arange(8, dtype=np.float32).reshape((1,4, 2))
    adj_matrix = np.array([[[1, 1, 0, 0],
                                [1, 1, 1, 1],
                                [0, 1, 1, 1],
                                [0, 1, 1, 1]]], dtype=np.float32)
    n_nodes = adj_matrix.shape[2]
    in_features = node_feats.shape[1]
    out_features = 2

    # W = np.random.uniform(size=(node_feats.shape[-1], out_features))
    W = np.array([[1., 0.], [0., 1.]])
    activation = np.tanh
    gcn = GCNLayer(n_nodes, in_features, out_features)

    print("Node features:\n", node_feats)
    print("\nAdjacency matrix:\n", adj_matrix)
    D_mod = np.zeros_like(adj_matrix)
    print(f"D_mod shape: {D_mod.shape}")
    # Filling it with the total number of neigh (plus self connections)
    np.fill_diagonal(D_mod[0], np.asarray(adj_matrix[0].sum(axis=1)).flatten())
    D_mod_invroot = np.linalg.inv(sqrtm(D_mod[0]))
    adj_matrix = D_mod_invroot @ adj_matrix @ D_mod_invroot
    print("\nAdjacency matrix:\n", adj_matrix)
    print("\n")
    neigh = adj_matrix.sum(axis=2, keepdims=True)
    print(neigh)
    neigh = adj_matrix.sum(axis=-1, keepdims=True)
    print(neigh)
    Z_temp = adj_matrix @ node_feats

    print("\n",Z_temp@W)