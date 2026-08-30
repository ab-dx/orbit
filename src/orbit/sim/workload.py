"""synthetic workload generator: poisson job arrivals over synthetic dags.

produces a list of (arrival_time, job_spec) pairs for an episode, where each
job_spec is a dict the env's _build_job understands (parallelism + stages with
tasks/duration/mu_cap/parents). deterministic for a given seed via
numpy.random.default_rng.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# every synthetic dag is sampled from one of these shapes; a shape builds a
# list of stage dicts and picks each stage's parent set to keep the graph
# acyclic.
TopologyFn = Callable[[int, "np.random.Generator"], list[dict]]


@dataclass
class WorkloadConfig:
    """tunables controlling the shape of a generated episode."""

    job_arrival_rate: float = 0.5       # poisson mean arrivals per second
    num_jobs: int = 10                  # how many jobs to emit
    cluster_size: int = 50              # caps job parallelism limits
    max_stages: int = 5                 # upper bound on stages per job
    max_tasks_per_stage: int = 50       # upper bound on tasks in a stage
    min_tasks_per_stage: int = 1
    max_avg_task_duration: float = 4.0  # upper bound on avg seconds per task
    min_avg_task_duration: float = 0.1
    topology: str = "mixed"             # chain, forkjoin, diamond, random, mixed

    def __post_init__(self) -> None:
        if self.topology not in ("chain", "forkjoin", "diamond", "random", "mixed"):
            raise ValueError(f"unknown topology: {self.topology}")


def poisson_arrivals(
    rate: float, num_jobs: int, rng: np.random.Generator
) -> np.ndarray:
    """sorted arrival times under a poisson (exponential inter-arrival) process."""
    if num_jobs <= 0:
        return np.array([], dtype=float)
    gaps: list[float] = []
    for _ in range(num_jobs):
        gaps.append(rng.exponential(1.0 / rate) if rate > 0 else 0.0)
    return np.cumsum(np.asarray(gaps, dtype=float))


def _topo_chain(n: int, rng) -> list[dict]:
    stages = []
    for i in range(n):
        stages.append({"parents": [i - 1] if i > 0 else []})
    return stages


def _topo_forkjoin(n: int, rng) -> list[dict]:
    stages = []
    if n == 1:
        return [{"parents": []}]
    stages.append({"parents": []})          # root
    for _ in range(n - 2):
        stages.append({"parents": [0]})     # parallel branches off the root
    stages.append({"parents": list(range(1, n - 1))})  # join
    return stages


def _topo_diamond(n: int, rng) -> list[dict]:
    # src -> <branch> -> sink
    if n == 1:
        return [{"parents": []}]
    stages = [{"parents": []}]
    for _ in range(n - 2):
        stages.append({"parents": [0]})
    stages.append({"parents": list(range(1, n - 1))})
    return stages


def _topo_random(n: int, rng) -> list[dict]:
    # each stage connects only to earlier stages, so it stays acyclic.
    stages = []
    for i in range(n):
        if i == 0:
            stages.append({"parents": []})
            continue
        possible = list(range(i))
        keep = rng.integers(0, len(possible) + 1)
        rng.shuffle(possible)
        parents = sorted(possible[: int(keep)])
        stages.append({"parents": list(parents)})
    return stages


_TOPOS: dict[str, tuple[TopologyFn, ...]] = {
    "chain": (_topo_chain,),
    "forkjoin": (_topo_forkjoin,),
    "diamond": (_topo_diamond,),
    "random": (_topo_random,),
}


def _stage_features(
    stages: list[dict], cfg: WorkloadConfig, rng: np.random.Generator
) -> list[dict]:
    """attach tasks and duration to each stage dict."""
    for st in stages:
        st["tasks"] = int(
            rng.integers(cfg.min_tasks_per_stage, cfg.max_tasks_per_stage + 1)
        )
        st["duration"] = float(
            rng.uniform(cfg.min_avg_task_duration, cfg.max_avg_task_duration)
        )
        st["mu_cap"] = 0.0
    return stages


def _sample_topology(
    n_stages: int, cfg: WorkloadConfig, rng: np.random.Generator
) -> list[dict]:
    if cfg.topology == "mixed":
        name = str(rng.choice(list(_TOPOS.keys())))
    else:
        name = cfg.topology
    builders = _TOPOS[name]
    fn = builders[int(rng.integers(0, len(builders)))]
    return fn(n_stages, rng)


def one_job(cfg: WorkloadConfig, rng: np.random.Generator) -> dict:
    """sample a single job spec from the workload distributions."""
    n_stages = int(rng.integers(1, cfg.max_stages + 1))
    stages = _sample_topology(n_stages, cfg, rng)
    stages = _stage_features(stages, cfg, rng)
    parallelism = int(rng.integers(1, cfg.cluster_size + 1))
    return {"parallelism": parallelism, "stages": stages}


def generate_episode(
    cfg: WorkloadConfig, seed: int | None = None
) -> list[tuple[float, dict]]:
    """sample an episode as (arrival_time, job_spec) pairs, arrival sorted."""
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(cfg.job_arrival_rate, cfg.num_jobs, rng)
    jobs: list[tuple[float, dict]] = []
    for t in arrivals:
        jobs.append((float(t), one_job(cfg, rng)))
    jobs.sort(key=lambda p: p[0])
    return jobs
