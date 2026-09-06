"""graph encoder: turns a decision-point observation into node embeddings.

a graph conv module that aggregates a stage's neighbors (parents and children
in its job dag) to build a context-aware embedding per node. the message
passing layer is written directly on the pyg MessagePassing base so the math
is explicit: each node's next hidden state is a nonlinearity over its own
state plus a mean over its neighbors' states, which share a weight.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MeanConv(MessagePassing):
    """a weight-shared mean-aggregate conv over directed dag edges.

    for each node i: h_i' = relu(w_self h_i + w_neigh * mean_{j in N(i)} h_j),
    where N(i) includes both parents and children, so information flows up
    and down the dag.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(aggr="mean")
        self.w_self = nn.Linear(in_dim, out_dim, bias=True)
        self.w_neigh = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # add the reverse edges so parents and children both pass messages
        symmetric = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        neigh_summary = self.propagate(symmetric, x=x)
        return F.relu(self.w_self(x) + self.w_neigh(neigh_summary))

    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        # pass raw neighbor features; aggr="mean" averages them before the
        # linear map, which equals mapping the averaged mean itself.
        return x_j


class GraphEncoder(nn.Module):
    """stacks an input embed with a few mean-conv layers into node embeddings."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList(
            [MeanConv(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """embed node features, returning one vector per node."""
        h = F.relu(self.embed(x))
        for conv in self.convs:
            h = conv(h, edge_index)
        return h
