import math
import numpy as np
import torch
import torch.nn as nn


zeroTolerance = 1e-9  # Values below this number are considered zero.
infiniteNumber = 1e12  # infinity equals this number


def LSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.
    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.
    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.
    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}
    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    assert S.shape[0] == E
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, 1, G, N])
    S = S.reshape([1, E, N, N])
    z = x.reshape([B, 1, 1, G, N]).repeat(1, E, 1, 1, 1)  # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1, K):
        x = torch.matmul(x, S)  # B x E x G x N
        xS = x.reshape([B, E, 1, G, N])  # B x E x 1 x G x N
        z = torch.cat((z, xS), dim=2)  # B x E x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x EKG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    z = z.permute(0, 4, 1, 2, 3).reshape([B, N, E * K * G]).double()
    h = h.reshape([F, E * K * G]).permute(1, 0).double()
    y = torch.matmul(z,h).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y


class GraphFilter(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter
    Initialization:
        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)
        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering
        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).
        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features
    Add graph shift operator:
        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).
        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                edge_features x number_nodes x number_nodes
    Forward call:
        y = GraphFilter(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes
        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G # in features
        self.F = F # out features
        self.K = K # filter taps
        self.E = E # edge features
        self.S = None  # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, E, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape E x N x N
        assert S.shape[0] == self.E
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N - Nin) \
                           .type(x.dtype).to(x.device)
                           ), dim=2)
        # Compute the filter output
        u = LSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
            self.G, self.F) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString



def batchLSIGF(h, SK, x, bias=None):
    """
    batchLSIGF(filter_taps, GSO_K, input, bias=None) Computes the output of a
        linear shift-invariant graph filter on input and then adds bias.
    In this case, we consider that there is a separate GSO to be used for each
    of the signals in the batch. In other words, SK[b] is applied when filtering
    x[b] as opposed to applying the same SK to all the graph signals in the
    batch.
    Inputs:
        filter_taps: vector of filter taps; size:
            output_features x edge_features x filter_taps x input_features
        GSO_K: collection of matrices; size:
            batch_size x edge_features x filter_taps x number_nodes x number_nodes
        input: input signal; size:
            batch_size x input_features x number_nodes
        bias: size: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}
    Outputs:
        output: filtered signals; size:
            batch_size x output_features x number_nodes
    """
    # Get the parameter numbers:
    F = h.shape[0]
    E = h.shape[1]
    K = h.shape[2]
    G = h.shape[3]
    B = SK.shape[0]
    assert SK.shape[1] == E
    assert SK.shape[2] == K
    N = SK.shape[3]
    assert SK.shape[4] == N
    assert x.shape[0] == B
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation I've been using:
    # h in F x E x K x G
    # SK in B x E x K x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N
    SK = SK.permute(1, 2, 0, 3, 4)
    # Now, SK is of shape E x K x B x N x N so that we can multiply by x of
    # size B x G x N to get
    z = torch.matmul(x, SK)
    # which is of size E x K x B x G x N.
    # Now, we have already carried out the multiplication across the dimension
    # of the nodes. Now we need to focus on the K, F, G.
    # Let's start by putting B and N in the front
    z = z.permute(2, 4, 0, 1, 3).reshape([B, N, E * K * G])
    # so that we get z in B x N x EKG.
    # Now adjust the filter taps so they are of the form EKG x F
    h = h.reshape([F, G * E * K]).permute(1, 0)
    # Multiply
    y = torch.matmul(z, h)
    # to get a result of size B x N x F. And permute
    y = y.permute(0, 2, 1)
    # to get it back in the right order: B x F x N.
    # Now, in this case, each element x[b,:,:] has adequately been filtered by
    # the GSO S[b,:,:,:]
    if bias is not None:
        y = y + bias
    return y

def matrixPowersBatch(S, K):
    """
    matrixPowers(A_b, K) Computes the matrix powers A_b^k for k = 0, ..., K-1
        for each A_b in b = 1, ..., B.
    Inputs:
        A (tensor): Matrices to compute powers. It can be either a single matrix
            per batch element: shape batch_size x number_nodes x number_nodes
            or contain edge features: shape
                batch_size x edge_features x number_nodes x number_nodes
        K (int): maximum power to be computed (up to K-1)
    Outputs:
        AK: either a collection of K matrices B x K x N x N (if the input was a
            single matrix) or a collection B x E x K x N x N (if the input was a
            collection of E matrices).
    """
    # S can be either a single GSO (N x N) or a collection of GSOs (E x N x N)
    if len(S.shape) == 3:
        B = S.shape[0]
        N = S.shape[1]
        assert S.shape[2] == N
        E = 1
        S = S.unsqueeze(1)
        scalarWeights = True
    elif len(S.shape) == 4:
        B = S.shape[0]
        E = S.shape[1]
        N = S.shape[2]
        assert S.shape[3] == N
        scalarWeights = False

    # Now, let's build the powers of S:
    thisSK = torch.eye(N).repeat([B, E, 1, 1]).to(S.device)
    SK = thisSK.unsqueeze(2)
    for k in range(1, K):
        thisSK = torch.matmul(thisSK, S)
        SK = torch.cat((SK, thisSK.unsqueeze(2)), dim=2)
    # Take out the first dimension if it was a single GSO
    if scalarWeights:
        SK = SK.squeeze(1)

    return SK

class GraphFilterBatchGSO(GraphFilter):
    """
    GraphFilterBatchGSO Creates a (linear) layer that applies a graph filter
        with a different GSO for each signal in the batch.
    This function is typically useful when not only the graph signal is changed
    during training, but also the GSO. That is, each data point in the batch is
    of the form (x_b,S_b) for b = 1,...,B instead of just x_b. The filter
    coefficients are still the same being applied to all graph filters, but both
    the GSO and the graph signal are different for each datapoint in the batch.
    Initialization:
        GraphFilterBatchGSO(in_features, out_features, filter_taps,
                            edge_features=1, bias=True)
        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering
        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).
        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features
    Add graph shift operator:
        GraphFilterBatchGSO.addGSO(GSO) Before applying the filter, we need to
        define the GSOs that we are going to use for each element of the batch.
        Each GSO has to have the same number of edges, but the number of nodes
        can change.
        Inputs:
            GSO (tensor): collection of graph shift operators; size can be
                batch_size x number_nodes x number_nodes, or
                batch_size x edge_features x number_nodes x number_nodes
    Forward call:
        y = GraphFilterBatchGSO(x)
        Inputs:
            x (tensor): input data; size: batch_size x in_features x number_nodes
        Outputs:
            y (tensor): output; size: batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, E=1, bias=True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__(G, F, K, E, bias)

    def addGSO(self, S):
        # So, we have to take into account the situation where S is either
        # B x N x N or B x E x N x N. No matter what, we're always handling,
        # internally the dimension E. So if the input is B x N x N, we have to
        # unsqueeze it so it becomes B x 1 x N x N.
        if len(S.shape) == 3 and S.shape[1] == S.shape[2]:
            self.S = S.unsqueeze(1)
        elif len(S.shape) == 4 and S.shape[1] == self.E \
                and S.shape[2] == S.shape[3]:
            self.S = S
        else:
            # TODO: print error
            pass

        self.N = self.S.shape[2]
        self.B = self.S.shape[0]
        self.SK = matrixPowersBatch(self.S, self.K)

    def forward(self, x):
        # TODO: If S (and consequently SK) hasn't been defined, print an error.
        return batchLSIGF(self.weight, self.SK, x, self.bias)

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
            self.G, self.F) + "filter_taps=%d, " % (
                         self.K) + "edge_features=%d, " % (self.E) + \
                     "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored: number_nodes=%d, batch_size=%d" % (
                self.N, self.B)
        else:
            reprString += "no GSO stored"
        return reprString

def BatchLSIGF(h, S, x, b=None):
    """
    LSIGF(filter_taps, GSO, input, bias=None) Computes the output of a linear
        shift-invariant graph filter on input and then adds bias.
    Denote as G the number of input features, F the number of output features,
    E the number of edge features, K the number of filter taps, N the number of
    nodes, S_{e} in R^{N x N} the GSO for edge feature e, x in R^{G x N} the
    input data where x_{g} in R^{N} is the graph signal representing feature
    g, and b in R^{F x N} the bias vector, with b_{f} in R^{N} representing the
    bias for feature f.
    Then, the LSI-GF is computed as
        y_{f} = \sum_{e=1}^{E}
                    \sum_{k=0}^{K-1}
                    \sum_{g=1}^{G}
                        [h_{f,g,e}]_{k} S_{e}^{k} x_{g}
                + b_{f}
    for f = 1, ..., F.
    Inputs:
        filter_taps (torch.tensor): array of filter taps; shape:
            output_features x edge_features x filter_taps x input_features
        GSO (torch.tensor): graph shift operator; shape:
            edge_features x number_nodes x number_nodes
        input (torch.tensor): input signal; shape:
            batch_size x input_features x number_nodes
        bias (torch.tensor): shape: output_features x number_nodes
            if the same bias is to be applied to all nodes, set number_nodes = 1
            so that b_{f} vector becomes b_{f} \mathbf{1}_{N}
    Outputs:
        output: filtered signals; shape:
            batch_size x output_features x number_nodes
    """
    # The basic idea of what follows is to start reshaping the input and the
    # GSO so the filter coefficients go just as a very plain and simple
    # linear operation, so that all the derivatives and stuff on them can be
    # easily computed.

    # h is output_features x edge_weights x filter_taps x input_features
    # S is edge_weighs x number_nodes x number_nodes
    # x is batch_size x input_features x number_nodes
    # b is output_features x number_nodes
    # Output:
    # y is batch_size x output_features x number_nodes

    # Get the parameter numbers:
    F = h.shape[0]
    K = h.shape[1]
    G = h.shape[2]
    N = S.shape[1]
    assert S.shape[2] == N
    B = x.shape[0]
    assert x.shape[1] == G
    assert x.shape[2] == N
    # Or, in the notation we've been using:
    # h in F x E x K x G
    # S in B x E x N x N
    # x in B x G x N
    # b in F x N
    # y in B x F x N

    # Now, we have x in B x G x N and S in B x E x N x N, and we want to come up
    # with matrix multiplication that yields z = x * S with shape
    # B x E x K x G x N.
    # For this, we first add the corresponding dimensions
    x = x.reshape([B, G, N])
    # print(S)
    S = S.reshape([B, N, N])
    z = x.reshape([B, 1, G, N]).repeat(1, 1, 1, 1) # This is for k = 0
    # We need to repeat along the E dimension, because for k=0, S_{e} = I for
    # all e, and therefore, the same signal values have to be used along all
    # edge feature dimensions.
    for k in range(1,K):
        x = torch.matmul(x, S.float()) # B x G x N
        xS = x.reshape([B, 1, G, N]) # B x 1 x G x N
        z = torch.cat((z, xS), dim = 2) # B x k x G x N
    # This output z is of size B x E x K x G x N
    # Now we have the x*S_{e}^{k} product, and we need to multiply with the
    # filter taps.
    # We multiply z on the left, and h on the right, the output is to be
    # B x N x F (the multiplication is not along the N dimension), so we reshape
    # z to be B x N x E x K x G and reshape it to B x N x KG (remember we
    # always reshape the last dimensions), and then make h be E x K x G x F and
    # reshape it to EKG x F, and then multiply
    y = torch.matmul(z.permute(0, 3, 1, 2).reshape([B, N, K*G]),
                     h.reshape([F, K*G]).permute(1, 0)).permute(0, 2, 1)
    # And permute againt to bring it from B x N x F to B x F x N.
    # Finally, add the bias
    if b is not None:
        y = y + b
    return y

class GraphFilterBatch(nn.Module):
    """
    GraphFilter Creates a (linear) layer that applies a graph filter
    Initialization:
        GraphFilter(in_features, out_features, filter_taps,
                    edge_features=1, bias=True)
        Inputs:
            in_features (int): number of input features (each feature is a graph
                signal)
            out_features (int): number of output features (each feature is a
                graph signal)
            filter_taps (int): number of filter taps
            edge_features (int): number of features over each edge
            bias (bool): add bias vector (one bias per feature) after graph
                filtering
        Output:
            torch.nn.Module for a graph filtering layer (also known as graph
            convolutional layer).
        Observation: Filter taps have shape
            out_features x edge_features x filter_taps x in_features
    Add graph shift operator:
        GraphFilter.addGSO(GSO) Before applying the filter, we need to define
        the GSO that we are going to use. This allows to change the GSO while
        using the same filtering coefficients (as long as the number of edge
        features is the same; but the number of nodes can change).
        Inputs:
            GSO (torch.tensor): graph shift operator; shape:
                Batch edge_features x number_nodes x number_nodes
    Forward call:
        y = GraphFilter(x)
        Inputs:
            x (torch.tensor): input data; shape:
                batch_size x in_features x number_nodes
        Outputs:
            y (torch.tensor): output; shape:
                batch_size x out_features x number_nodes
    """

    def __init__(self, G, F, K, bias = True):
        # K: Number of filter taps
        # GSOs will be added later.
        # This combines both weight scalars and weight vectors.
        # Bias will always be shared and scalar.

        # Initialize parent
        super().__init__()
        # Save parameters:
        self.G = G
        self.F = F
        self.K = K
        self.S = None # No GSO assigned yet
        # Create parameters:
        self.weight = nn.parameter.Parameter(torch.Tensor(F, K, G))
        if bias:
            self.bias = nn.parameter.Parameter(torch.Tensor(F, 1))
        else:
            self.register_parameter('bias', None)
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Taken from _ConvNd initialization of parameters:
        stdv = 1. / math.sqrt(self.G * self.K)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def addGSO(self, S):
        # Every S has 3 dimensions.
        assert len(S.shape) == 3
        # S is of shape B x N x N
        self.N = S.shape[1]
        assert S.shape[2] == self.N
        self.S = S

    def forward(self, x):
        # x is of shape: batchSize x dimInFeatures x numberNodesIn
        B = x.shape[0]
        F = x.shape[1]
        Nin = x.shape[2]
        # And now we add the zero padding
        if Nin < self.N:
            x = torch.cat((x,
                           torch.zeros(B, F, self.N-Nin)\
                                   .type(x.dtype).to(x.device)
                          ), dim = 2)
        # Compute the filter output
        u = BatchLSIGF(self.weight, self.S, x, self.bias)
        # So far, u is of shape batchSize x dimOutFeatures x numberNodes
        # And we want to return a tensor of shape
        # batchSize x dimOutFeatures x numberNodesIn
        # since the nodes between numberNodesIn and numberNodes are not required
        if Nin < self.N:
            u = torch.index_select(u, 2, torch.arange(Nin).to(u.device))
        return u

    def extra_repr(self):
        reprString = "in_features=%d, out_features=%d, " % (
                        self.G, self.F) + "filter_taps=%d, " % (
                        self.K) + \
                        "bias=%s, " % (self.bias is not None)
        if self.S is not None:
            reprString += "GSO stored"
        else:
            reprString += "no GSO stored"
        return reprString