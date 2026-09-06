"""the grpc scheduler servicer, doing greedy inference with a loaded policy."""

from __future__ import annotations

import torch

from ..agent import Policy
from . import scheduler_pb2, scheduler_pb2_grpc


class SchedulerServicer(scheduler_pb2_grpc.SchedulerServicer):
    """answers decide requests with the policy's greedy action."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def Decide(self, request, context) -> scheduler_pb2.DecideResponse:
        x = torch.tensor(request.x, dtype=torch.float32).reshape(
            -1, request.n_features
        )
        edge_index = torch.tensor(
            [list(request.edge_src), list(request.edge_dst)], dtype=torch.long
        )
        runnable = list(request.runnable_node_ids)
        available = list(request.available)
        with torch.no_grad():
            h = self.policy.encode(x, edge_index)
            action = self.policy.act(h, runnable, available)
        return scheduler_pb2.DecideResponse(
            stage_action=action.stage,
            alloc_action=action.alloc,
            stage_node_id=runnable[action.stage],
        )
