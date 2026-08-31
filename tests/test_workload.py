"""milestone 1.6: synthetic workload generator + time-based arrivals."""

from __future__ import annotations

import pytest

from orbit import core
from orbit.sim import OrbitEnv, WorkloadConfig, generate_episode


def _cfg(**kw):
    defaults = dict(
        job_arrival_rate=1.0,
        num_jobs=6,
        cluster_size=8,
        max_stages=4,
        min_tasks_per_stage=1,
        max_tasks_per_stage=6,
        min_avg_task_duration=0.5,
        max_avg_task_duration=2.5,
    )
    defaults.update(kw)
    return WorkloadConfig(**defaults)


def _assert_valid_dag(ep):
    for _t, spec in ep:
        n = len(spec["stages"])
        assert 1 <= n
        assert 1 <= spec["parallelism"] <= 8
        for i, st in enumerate(spec["stages"]):
            # every parent edge points at an earlier stage, so the graph is
            # acyclic and stays topologically sorted
            for p in st.get("parents", []):
                assert 0 <= p < i
            assert 1 <= st["tasks"]
            assert 0.5 <= st["duration"] <= 2.5
            assert st["mu_cap"] == 0.0


def test_generation_is_deterministic_per_seed() -> None:
    a = generate_episode(_cfg(), seed=7)
    b = generate_episode(_cfg(), seed=7)
    c = generate_episode(_cfg(), seed=8)
    assert a == b
    assert a != c


def test_arrivals_sorted_and_counted() -> None:
    ep = generate_episode(_cfg(num_jobs=10), seed=3)
    assert len(ep) == 10
    times = [t for t, _ in ep]
    assert times == sorted(times)
    assert all(t >= 0.0 for t in times)


def test_all_topologies_produce_valid_dags() -> None:
    for topo in ("chain", "forkjoin", "diamond", "random", "mixed"):
        ep = generate_episode(_cfg(num_jobs=200, topology=topo), seed=11)
        _assert_valid_dag(ep)


def test_chain_topology_is_sequential() -> None:
    ep = generate_episode(_cfg(num_jobs=1, topology="chain", max_stages=4), seed=5)
    _spec = ep[0][1]
    for i, st in enumerate(_spec["stages"]):
        if i == 0:
            assert st["parents"] == []
        else:
            assert st["parents"] == [i - 1]


def test_root_and_join_shapes() -> None:
    # forkjoin and diamond both open with a root and close with a join
    for topo in ("forkjoin", "diamond"):
        found_fj = False
        for _t, spec in generate_episode(
            _cfg(num_jobs=50, topology=topo, max_stages=6), seed=9
        ):
            if len(spec["stages"]) >= 2:
                assert spec["stages"][0]["parents"] == []
            if len(spec["stages"]) >= 3:
                assert len(spec["stages"][-1]["parents"]) >= 1
                found_fj = True
        assert found_fj


def test_workload_config_rejects_unknown_topology() -> None:
    with pytest.raises(ValueError):
        WorkloadConfig(topology="nope")


def _run_episode(seed):
    env = OrbitEnv(core.SimulatorConfig())
    obs, info = env.reset(options={"workload": _cfg(), "seed": seed})
    total = 0.0
    steps = 0
    terminated = False
    while not terminated:
        obs, reward, terminated, _trunc, info = env.step((0, 1))  # tile -> 2 executors
        total += reward
        steps += 1
        assert steps < 500
    return env, total, steps, info


def test_env_drives_full_workload_episode() -> None:
    env, total, steps, info = _run_episode(seed=4)
    assert info["terminated"] is True
    assert info["arrived"] == _cfg().num_jobs
    assert total < 0.0  # all jcts accumulate as negative reward
    # every job added over the episode eventually completed
    assert all(env._sim.job_done(j) for j in range(len(env._sim.jobs())))


def test_arrivals_stagger_and_grow_runnable_set() -> None:
    env = OrbitEnv(core.SimulatorConfig())
    obs, info = env.reset(options={"workload": _cfg(num_jobs=4), "seed": 1})
    # no arrivals yet; time has not advanced and nothing is runnable
    assert info["arrived"] == 0
    assert obs.runnable == []

    arrived_seen = []
    terminated = False
    for _ in range(60):
        if terminated:
            break
        obs, _r, terminated, _trunc, info = env.step((0, 1))
        if info["arrived"] not in arrived_seen:
            arrived_seen.append(info["arrived"])
    assert arrived_seen[0] > 0  # the first step advanced to an arrival
    assert len(arrived_seen) > 1  # arrivals landed at different times


def test_episode_with_explicit_jobs_still_immediate() -> None:
    # the all-at-t0 path is unchanged: jobs visible at reset
    env = OrbitEnv(core.SimulatorConfig())
    obs, info = env.reset(
        options={"jobs": [{"parallelism": 2, "stages": [{"tasks": 2}]}]}
    )
    assert info["arrived"] == 1
    assert info["runnable"] == [(0, 0)]
