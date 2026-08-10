"""
Does the policy treat the fleet as an unordered set?

A fleet has no canonical ordering. Relabelling the robots - permuting the
observations and the adjacency matrix identically - must permute the outputs and
change nothing else. Every weight in these models is shared across robots, so
this should hold by construction.

It did not, for the message-passing variant, for the whole life of the project.
The self-term reinterpreted a (B, F, N) buffer as (B, N, F) instead of
transposing it, which made each robot's "own" features a fixed stride through
the entire fleet's. The result was a global channel: information reached a robot
from robots it had no edge to. See docs/findings.md.

Nothing else in the suite could have caught it. Every other test asks whether a
number improved, and a policy given extra information improves.
"""

import torch

from models.framework_gnn import Network as GCNNetwork
from models.networks.gnn import GCNLayer, MessagePassingLayer

FLEET_SIZES = (4, 10, 33, 64, 100)


def make_config(agents):
    return {
        "device": "cpu",
        "map_shape": [5, 5],
        "num_agents": agents,
        "num_actions": 5,
        "last_convs": [400],
        "channels": [16, 16, 16],
        "encoder_layers": 1,
        "encoder_dims": [64],
        "graph_filters": [3],
        "node_dims": [128],
        "action_layers": 1,
    }


def ring_adjacency(agents):
    """A connected, non-trivial graph. A complete or empty one hides mistakes."""
    adj = torch.zeros(1, agents, agents)
    for i in range(agents):
        adj[0, i, (i + 1) % agents] = 1.0
        adj[0, (i + 1) % agents, i] = 1.0
    return adj


def permutation_error(net, agents, seed=0):
    """Max |f(P x)[P^-1] - f(x)| — zero iff the network is permutation-equivariant."""
    torch.manual_seed(seed)
    states = torch.randn(1, agents, 2, 5, 5)
    adj = ring_adjacency(agents)

    perm = torch.randperm(agents)
    inverse = torch.argsort(perm)

    net.eval()
    with torch.no_grad():
        plain = net(states, adj)
        permuted = net(states[:, perm], adj[:, perm][:, :, perm])
    return (permuted[:, inverse] - plain).abs().max().item()


def test_gcn_is_permutation_equivariant():
    for agents in FLEET_SIZES:
        net = GCNNetwork(make_config(agents))
        error = permutation_error(net, agents)
        assert error < 1e-4, f"GCN is order-dependent at {agents} agents: {error}"


def test_graph_layers_do_not_mix_features_across_the_fleet():
    """
    The layer alone, without the CNN in front of it.

    Isolating it means a failure points at the aggregation rather than anywhere
    in the encoder, and it is where the defect lived.
    """
    agents, in_features, out_features = 12, 64, 128
    torch.manual_seed(0)

    for layer in (
        GCNLayer(agents, in_features, out_features, filter_number=3),
        MessagePassingLayer(agents, in_features, out_features, filter_number=3),
    ):
        layer.eval()
        # The framework hands these layers (B, F, N).
        feats = torch.randn(1, in_features, agents)
        adj = ring_adjacency(agents)
        perm = torch.randperm(agents)
        inverse = torch.argsort(perm)

        layer.addGSO(adj)
        with torch.no_grad():
            plain = layer(feats)
        layer.addGSO(adj[:, perm][:, :, perm])
        with torch.no_grad():
            permuted = layer(feats[:, :, perm])

        error = (permuted[:, :, inverse] - plain).abs().max().item()
        assert error < 1e-4, f"{type(layer).__name__} is order-dependent: {error}"


def test_disconnected_agents_cannot_influence_each_other():
    """
    With an empty communication graph a robot must be unreachable.

    This is the property the demo depends on: a decentralised policy cannot read
    a robot it has no edge to, however crowded the board gets.
    """
    agents = 8
    net = GCNNetwork(make_config(agents))
    net.eval()

    torch.manual_seed(0)
    states = torch.randn(1, agents, 2, 5, 5)
    empty = torch.zeros(1, agents, agents)

    disturbed = states.clone()
    disturbed[0, agents - 1] = torch.randn(2, 5, 5)

    with torch.no_grad():
        before = net(states, empty)[0, 0]
        after = net(disturbed, empty)[0, 0]

    error = (after - before).abs().max().item()
    assert error < 1e-5, f"an unconnected agent changed another's logits by {error}"
