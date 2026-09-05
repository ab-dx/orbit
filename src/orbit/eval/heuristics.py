"""deterministic scheduling heuristics for head to head eval."""

from __future__ import annotations

from ..sim import ALLOC_TILES


def _greedy_tile(available: int) -> int:
    """largest tile index that fits the available executors."""
    tile = 0
    for i, size in enumerate(ALLOC_TILES):
        if size <= available:
            tile = i
        else:
            break
    return tile


def _tile_at_most(count: int) -> int:
    """largest tile index no larger than count, or the no-op tile."""
    tile = 0
    for i, size in enumerate(ALLOC_TILES):
        if size <= count:
            tile = i
        else:
            break
    return tile


def fifo(obs, info: dict) -> tuple[int, int]:
    """first runnable stage with a greedy grant."""
    if not info["available"] or info["available"][0] <= 0:
        return (0, 0)
    return (0, _greedy_tile(info["available"][0]))


def sjf(obs, info: dict) -> tuple[int, int]:
    """runnable stage of the job with the least remaining work."""
    remaining: dict[int, float] = {}
    for n in obs.nodes:
        if n.tasks_remaining > 0:
            remaining[n.job_index] = remaining.get(n.job_index, 0.0) + (
                n.tasks_remaining * n.avg_task_duration
            )
    if not info["runnable"]:
        return (0, 0)
    runnable = info["runnable"]
    best = 0
    best_work = float("inf")
    for i, (job_idx, _stage) in enumerate(runnable):
        work = remaining.get(job_idx, float("inf"))
        if work < best_work:
            best, best_work = i, work
    available = info["available"][best]
    if available <= 0:
        return (0, 0)
    return (best, _greedy_tile(available))


def _share(job_available: int, n_jobs: int, weight: float, weights: float) -> int:
    """fair share of a job's capacity, clamped to at least one executor."""
    if job_available <= 0:
        return 0
    if weights <= 0:
        share = job_available / n_jobs
    else:
        share = job_available * weight / weights
    share = max(1.0, share)
    return int(min(share, job_available))


def fair(obs, info: dict) -> tuple[int, int]:
    """serve runnable jobs in order with an equal slice of capacity."""
    available = info["available"]
    n = len(available)
    if n == 0:
        return (0, 0)
    target = _share(available[0], n, 1.0, float(n))
    if target <= 0:
        return (0, 0)
    return (0, _tile_at_most(target))


def weighted_fair(obs, info: dict) -> tuple[int, int]:
    """serve runnable jobs in order, slicing capacity by job weight."""
    if not info["runnable"]:
        return (0, 0)
    # weight by the owning job parallelism limit
    limit_of: dict[int, int] = {}
    for n in obs.nodes:
        limit_of[n.job_index] = n.parallelism_limit
    job0 = info["runnable"][0][0]
    weight = float(limit_of.get(job0, 1))
    weights = sum(float(limit_of.get(j, 1)) for j, _s in info["runnable"])
    available = info["available"]
    if not available or available[0] <= 0:
        return (0, 0)
    target = _share(available[0], len(available), weight, weights)
    if target <= 0:
        return (0, 0)
    return (0, _tile_at_most(target))


HEURISTICS: dict[str, object] = {
    "fifo": fifo,
    "sjf": sjf,
    "fair": fair,
    "weighted_fair": weighted_fair,
}


__all__ = [
    "HEURISTICS",
    "fair",
    "fifo",
    "sjf",
    "weighted_fair",
]
