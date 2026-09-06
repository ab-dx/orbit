"""a grpc client for the scheduler service."""

from __future__ import annotations

from ..sim.observe import FEATURE_COLUMNS, node_features
from . import scheduler_pb2, scheduler_pb2_grpc


def build_request(obs, info: dict) -> scheduler_pb2.DecideRequest:
    """pack an env observation plus info dict into a decide request."""
    return scheduler_pb2.DecideRequest(
        x=[float(v) for v in node_features(obs).ravel()],
        n_features=len(FEATURE_COLUMNS),
        edge_src=list(obs.edge_src),
        edge_dst=list(obs.edge_dst),
        runnable_node_ids=list(info["runnable_node_ids"]),
        available=list(info["available"]),
    )


class SchedulerClient:
    """a thin wrapper over the scheduler stub."""

    def __init__(self, channel) -> None:
        self.stub = scheduler_pb2_grpc.SchedulerStub(channel)

    def decide(self, request: scheduler_pb2.DecideRequest) -> scheduler_pb2.DecideResponse:
        return self.stub.Decide(request)


__all__ = ["SchedulerClient", "build_request"]
