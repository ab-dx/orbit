"""curriculum episode length.

ramps the number of jobs in an episode from a small start up to the target as
training progresses, so the policy first learns on easy short workloads and
gets harder ones gradually.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..sim import WorkloadConfig


@dataclass
class Curriculum:
    """controls how episode size grows over training."""

    start_jobs: int = 2        # job count at iteration zero
    ramp_iters: int = 50       # iters to go from start to target
    enabled: bool = False


def wcfg_for(curriculum: Curriculum, it: int, target: WorkloadConfig) -> WorkloadConfig:
    """a copy of target with the job count ramped for iteration it."""
    if not curriculum.enabled:
        return target
    start = curriculum.start_jobs
    step = (target.num_jobs - start) / max(curriculum.ramp_iters, 1)
    num = int(round(start + step * min(it, curriculum.ramp_iters)))
    num = max(start, min(num, target.num_jobs))
    return WorkloadConfig(
        job_arrival_rate=target.job_arrival_rate,
        num_jobs=num,
        cluster_size=target.cluster_size,
        max_stages=target.max_stages,
        max_tasks_per_stage=target.max_tasks_per_stage,
        min_tasks_per_stage=target.min_tasks_per_stage,
        max_avg_task_duration=target.max_avg_task_duration,
        min_avg_task_duration=target.min_avg_task_duration,
        topology=target.topology,
    )


__all__ = ["Curriculum", "wcfg_for"]
