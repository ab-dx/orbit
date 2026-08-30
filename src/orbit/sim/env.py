"""a gymnasium environment wrapping the c++ simulator for rl.

each step chooses one scheduling decision: which runnable stage to give
executors to, and how many (from a fixed set of allocation tiles). the
simulator then runs autonomously until the next decision point, where the
env presents a fresh observation and whatever job completion time accrued
since the last step as a negative reward.

jobs arrive over time: reset either takes explicit job specs (all arriving at
t=0) or a workload config, in which case arrivals follow a poisson process via
the workload generator.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
from gymnasium import spaces

from orbit import core

from .workload import WorkloadConfig, generate_episode

# fixed executor-grant sizes the policy can choose between, in ascending
# order. the chosen tile is clamped to the selected job's available capacity.
ALLOC_TILES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


class OrbitEnv(gym.Env):
    """drive the c++ simulator through scheduling decisions.

    reset takes options with either:
      jobs:   a list of job specs (all arrive at t=0), or
      workload: a WorkloadConfig, from which an episode is generated with
                poisson arrivals spread over time.
    each job spec is a dict with 'parallelism' and 'stages', where each stage
    is a dict with 'tasks', 'parents', 'duration', 'mu_cap'.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: core.SimulatorConfig | None = None) -> None:
        super().__init__()
        self._cfg = cfg or core.SimulatorConfig()
        self._sim: core.Simulator | None = None
        self._n_total_stages = 0
        self._n_jobs = 0
        self._arrived: list[bool] = []
        # a single flat discrete action: flat = stage_idx * n_tiles + tile_idx.
        # stage_idx indexes into the runnable set; tile_idx into ALLOC_TILES.
        self.action_space = spaces.Discrete(1)
        # observation is returned as a raw core.Observation; documented here.
        self.observation_space = spaces.Dict({})

    # job construction

    @staticmethod
    def _build_job(spec: dict) -> core.Job:
        job = core.Job()
        job.parallelism_limit = int(spec.get("parallelism", 1))
        stages = []
        for s in spec.get("stages", []):
            st = core.Stage()
            st.total_tasks = int(s.get("tasks", 0))
            st.tasks_remaining = int(s.get("tasks", 0))
            st.avg_task_duration = float(s.get("duration", 1.0))
            st.mu_cap = float(s.get("mu_cap", 0.0))
            st.parents = list(s.get("parents", []))
            stages.append(st)
        job.stages = stages
        return job

    # gymnasium api

    def reset(
        self, *, seed=None, options: dict | None = None
    ) -> tuple[core.Observation, dict]:
        """clear the simulator and load an episode."""
        super().reset(seed=seed)
        options = options or {}

        self._sim = core.Simulator(self._cfg)
        self._n_total_stages = 0
        self._n_jobs = 0
        self._arrived = []

        plan = self._plan(options)
        for idx, (arrive_at, spec) in enumerate(plan):
            job = self._build_job(spec)
            self._n_total_stages += len(job.stages)
            if arrive_at <= 0.0:
                # insert immediately so the job is visible at reset
                self._sim.add_job(job)
                self._arrived[idx] = True
            else:
                self._sim.add_job_at(job, arrive_at)

        self.action_space = spaces.Discrete(self._n_total_stages * len(ALLOC_TILES))

        obs = self._sim.observe()
        return obs, {
            "runnable": list(obs.runnable),
            "terminated": False,
            "arrived": sum(self._arrived),
        }

    def step(
        self, action: int
    ) -> tuple[core.Observation, float, bool, bool, dict]:
        """apply one scheduling decision, then run to the next decision point."""
        sim = self._sim
        assert sim is not None, "call reset before step"

        runnable = list(sim.observe().runnable)

        if runnable:
            flat = int(action)
            tile_idx = flat % len(ALLOC_TILES)
            stage_idx = flat // len(ALLOC_TILES)
            if stage_idx < len(runnable):
                job_idx, _stage = runnable[stage_idx]
                count = min(ALLOC_TILES[tile_idx], sim.available_for(job_idx))
                if count > 0:
                    sim.add_executors(job_idx, count)

        reward = -self._advance()

        obs = sim.observe()
        self._reconcile_arrivals()
        terminated = self._all_done()
        info = {
            "runnable": list(obs.runnable),
            "terminated": terminated,
            "time": sim.now(),
            "arrived": sum(self._arrived),
        }
        return obs, float(reward), terminated, False, info

    # workload planning

    def _plan(self, options: dict) -> list[tuple[float, dict]]:
        """resolve the episode's (arrival_time, job_spec) plan."""
        if "workload" in options:
            wcfg = options["workload"]
            if not isinstance(wcfg, WorkloadConfig):
                wcfg = WorkloadConfig(**wcfg)
            seed = options.get("seed")
            plan = generate_episode(wcfg, seed=seed)
            self._n_jobs = len(plan)
            self._arrived = [False] * self._n_jobs
            return plan
        specs: Sequence[dict] = options.get("jobs", [])
        self._n_jobs = len(specs)
        self._arrived = [False] * self._n_jobs
        return [(0.0, spec) for spec in specs]

    # internals

    def _reconcile_arrivals(self) -> None:
        """mark dispatched indices as arrived once their stage data is in."""
        sim = self._sim
        if sim is None or not self._arrived:
            return
        for i, arrived in enumerate(self._arrived):
            if not arrived and i < len(sim.jobs()):
                if len(sim.jobs()[i].stages) > 0:
                    self._arrived[i] = True

    def _all_done(self) -> bool:
        sim = self._sim
        if sim is None:
            return False
        if not self._arrived:
            return False
        for i, arrived in enumerate(self._arrived):
            if not arrived:
                return False
            if not sim.job_done(i):
                return False
        return True

    def _advance(self) -> float:
        """run the simulator to the next decision point.

        advances step by step, accumulating positive jct as a negative reward.
        stops when the event queue empties or the runnable set changes, i.e.
        when a fresh allocation decision is relevant again. a job arrival adds
        a runnable root, so it naturally ends the advance.
        """
        sim = self._sim
        assert sim is not None
        jct = 0.0
        while True:
            runnable_before = set(sim.observe().runnable)
            if not sim.step():
                break
            jct += sim.jct_since_step()
            runnable_after = set(sim.observe().runnable)
            if runnable_before != runnable_after:
                break
        return jct
