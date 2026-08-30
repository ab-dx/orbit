"""a gymnasium environment wrapping the c++ simulator for rl.

each step chooses one scheduling decision: which runnable stage to give
executors to, and how many (from a fixed set of allocation tiles). the
simulator then runs autonomously until the next decision point, where the
env presents a fresh observation and whatever job completion time accrued
since the last step as a negative reward.
"""

from __future__ import annotations

from typing import Sequence

import gymnasium as gym
from gymnasium import spaces

from orbit import core

# fixed executor-grant sizes the policy can choose between, in ascending
# order. the chosen tile is clamped to the selected job's available capacity.
ALLOC_TILES: tuple[int, ...] = (1, 2, 4, 8, 16, 32)


class OrbitEnv(gym.Env):
    """drive the c++ simulator through scheduling decisions.

    reset takes a list of job specs. each spec is a dict with 'parallelism'
    and 'stages', where each stage is a dict with 'tasks', 'parents',
    'duration', 'mu_cap'. jobs are added up front and all run to completion
    within the episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: core.SimulatorConfig | None = None) -> None:
        super().__init__()
        self._cfg = cfg or core.SimulatorConfig()
        self._sim: core.Simulator | None = None
        self._n_total_stages = 0
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
        """clear the simulator and load an episode's job list."""
        specs: Sequence[dict] = options.get("jobs", []) if options else []
        super().reset(seed=seed)

        self._sim = core.Simulator(self._cfg)
        self._n_total_stages = 0
        for spec in specs:
            job = self._build_job(spec)
            self._n_total_stages += len(job.stages)
            self._sim.add_job(job)

        # a single flat discrete action codes both heads: the runnable stage
        # index and the allocation tile index.
        self.action_space = spaces.Discrete(self._n_total_stages * len(ALLOC_TILES))

        obs = self._sim.observe()
        return obs, {"runnable": list(obs.runnable), "terminated": False}

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
        terminated = self._all_done()
        info = {
            "runnable": list(obs.runnable),
            "terminated": terminated,
            "time": sim.now(),
        }
        return obs, float(reward), terminated, False, info

    # internals

    def _all_done(self) -> bool:
        sim = self._sim
        return sim is not None and all(sim.job_done(j) for j in range(len(sim.jobs())))

    def _advance(self) -> float:
        """run the simulator to the next decision point.

        advances step by step, accumulating positive jct as a negative reward.
        stops when the event queue empties or the runnable set changes, i.e.
        when a fresh allocation decision is relevant again.
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
