"""torch/pyg-agnostic helpers to reshape a c++ observation for the graph
encoder. keeps the ml stack decoupled from the simulator until phase 3.
"""

from __future__ import annotations

import numpy as np

# feature order of each row in node_features(); keep in sync with StageObs
# fields exposed by the c++ core.
FEATURE_COLUMNS = [
    "tasks_remaining",
    "avg_task_duration",
    "assigned_executors",
    "mu_cap",
    "parallelism_limit",
    "headroom",
    "wave_count",
    "completed",
    "is_root",
    "is_leaf",
    "runnable",
    "age",
]

_NODE_FEATURE_GETTERS = [
    lambda s: float(s.tasks_remaining),
    lambda s: float(s.avg_task_duration),
    lambda s: float(s.assigned_executors),
    lambda s: float(s.mu_cap),
    lambda s: float(s.parallelism_limit),
    lambda s: float(s.headroom),
    lambda s: float(s.wave_count),
    lambda s: float(s.completed),
    lambda s: float(s.is_root),
    lambda s: float(s.is_leaf),
    lambda s: float(s.runnable),
    lambda s: float(s.age),
]


def node_features(obs) -> np.ndarray:
    """matrix of shape (n_nodes, n_features) as float32."""
    rows = [[g(s) for g in _NODE_FEATURE_GETTERS] for s in obs.nodes]
    return np.asarray(rows, dtype=np.float32)


def edge_index(obs) -> np.ndarray:
    """edge index of shape (2, n_edges); each column is (src, dst) node id."""
    src = np.asarray(obs.edge_src, dtype=np.int64)
    dst = np.asarray(obs.edge_dst, dtype=np.int64)
    return np.stack([src, dst])


def runnable_set(obs) -> list[tuple[int, int]]:
    """the (job_index, stage_index) pairs a policy may schedule."""
    return list(obs.runnable)
