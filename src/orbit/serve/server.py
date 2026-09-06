"""build and run the grpc scheduling server."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import grpc

from ..agent import Policy
from . import scheduler_pb2_grpc
from .service import SchedulerServicer


def serve(
    policy: Policy,
    port: int = 50051,
    max_workers: int = 4,
) -> tuple[grpc.Server, int]:
    """bind a scheduler server on the given port, ready to start."""
    server = grpc.server(ThreadPoolExecutor(max_workers=max_workers))
    scheduler_pb2_grpc.add_SchedulerServicer_to_server(
        SchedulerServicer(policy), server
    )
    bound = server.add_insecure_port(f"0.0.0.0:{port}")
    return server, bound


__all__ = ["serve"]
