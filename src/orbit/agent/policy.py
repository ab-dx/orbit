"""a two-headed policy over the scheduling action space.

one head picks which runnable stage to grant executors to; the other picks how
many executors (an allocation tile), conditioned on the chosen stage. both
heads are masked so only legal actions can be sampled, which is what make
reinforce well behaved: the policy never proposes a stage that is not runnable
or an allocation larger than the pool and the job allow.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from ..sim import ALLOC_TILES
from ..sim.observe import edge_index, node_features
from .encoder import GraphEncoder


def graph_input(obs) -> tuple[torch.Tensor, torch.Tensor]:
    """convert a c++ observation into (x, edge_index) tensors for the encoder."""
    x = torch.tensor(node_features(obs), dtype=torch.float32)
    ei = torch.tensor(edge_index(obs), dtype=torch.long)
    return x, ei


@dataclass
class Action:
    """one scheduling decision: a runnable stage index and an alloc tile."""
    stage: int
    # index into the alloc heads's tile list; the trailing no-op tile grants 0
    alloc: int


# index of the no-op "wait, grant nothing" option in the alloc head.
NOOP_ALLOC = len(ALLOC_TILES)


class Policy(torch.nn.Module):
    """graph encoder plus stage and allocation policy heads."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        tiles: tuple[int, ...] = ALLOC_TILES,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(in_dim, hidden_dim, num_layers)
        self.stage_head = torch.nn.Linear(hidden_dim, 1)
        # one extra unit for the no-op wait option that grants zero executors
        self.alloc_head = torch.nn.Linear(hidden_dim, len(tiles) + 1)
        self.tiles = list(tiles) + [0]
        self.n_tiles = len(self.tiles)

    # encoding helpers

    def encode(self, x, edge_index) -> torch.Tensor:
        return self.encoder(x, edge_index)

    # head computations

    def stage_distribution(self, h, runnable_ids: list[int]):
        """categorical over the runnable stages given their node embeddings."""
        h_r = h[torch.as_tensor(runnable_ids, dtype=torch.long)]
        scores = self.stage_head(h_r).squeeze(-1)
        return Categorical(logits=scores)

    def alloc_distribution(self, h, stage_action: int, available: int, runnable_ids: list[int]):
        """categorical over allocation tiles, masked to what is available."""
        node_id = runnable_ids[stage_action]
        logits = self.alloc_head(h[node_id])
        # a tile is legal if it is the no-op (grants 0) or fits the available
        # executors; anything larger than available is impossible to grant.
        tile_mask = torch.as_tensor(
            [t == 0 or t <= available for t in self.tiles],
            dtype=torch.bool,
        )
        logits = logits.masked_fill(~tile_mask, float("-inf"))
        return Categorical(logits=logits)

    # action interface

    def sample(self, h, runnable_ids: list[int], available: list[int]) -> Action:
        """sample a legal (stage, alloc) action; assumes runnable is non-empty."""
        stage_dist = self.stage_distribution(h, runnable_ids)
        stage_action = int(stage_dist.sample().item())
        alloc_dist = self.alloc_distribution(h, stage_action, available[stage_action], runnable_ids)
        alloc_action = int(alloc_dist.sample().item())
        return Action(stage=stage_action, alloc=alloc_action)

    def act(self, h, runnable_ids: list[int], available: list[int]) -> Action:
        """the greedy best (stage, alloc) action; for deterministic serving."""
        stage_dist = self.stage_distribution(h, runnable_ids)
        stage_action = int(stage_dist.logits.argmax().item())
        alloc_dist = self.alloc_distribution(h, stage_action, available[stage_action], runnable_ids)
        alloc_action = int(alloc_dist.logits.argmax().item())
        return Action(stage=stage_action, alloc=alloc_action)

    def log_prob(
        self,
        h,
        runnable_ids: list[int],
        available: list[int],
        action: Action,
    ) -> torch.Tensor:
        """log probability of a (stage, alloc) action under the heads."""
        stage_dist = self.stage_distribution(h, runnable_ids)
        alloc_dist = self.alloc_distribution(h, action.stage, available[action.stage], runnable_ids)
        lp = stage_dist.log_prob(torch.tensor(action.stage)) + alloc_dist.log_prob(
            torch.tensor(action.alloc)
        )
        return lp

