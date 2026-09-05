from __future__ import annotations

import torch

from orbit import core
from orbit.agent import Policy
from orbit.eval import (
    HEURISTICS,
    evaluate,
    policy_decide,
    run_scheduler,
    summarize,
)
from orbit.sim import ALLOC_TILES, OrbitEnv, WorkloadConfig
from orbit.train import baseline_rollout

N_FEATURES = 12


def _env() -> OrbitEnv:
    return OrbitEnv(core.SimulatorConfig())


def _wcfg(**kw) -> WorkloadConfig:
    defaults = dict(
        job_arrival_rate=1.0,
        num_jobs=3,
        cluster_size=8,
        max_stages=3,
        min_tasks_per_stage=1,
        max_tasks_per_stage=4,
        min_avg_task_duration=0.5,
        max_avg_task_duration=1.5,
    )
    defaults.update(kw)
    return WorkloadConfig(**defaults)


def test_heuristics_return_legal_actions() -> None:
    env = _env()
    wcfg = _wcfg()
    for name, decide in HEURISTICS.items():
        obs, info = env.reset(options={"workload": wcfg, "seed": 0})
        # step until runnable appears, then check the decision
        for _ in range(20):
            if info["runnable"]:
                break
            obs, _r, _t, _tr, info = env.step(decide(obs, info))
        assert info["runnable"], f"{name} never became runnable"
        stage, alloc = decide(obs, info)
        assert 0 <= stage < len(info["runnable"])
        if info["available"][stage] > 0:
            assert 0 <= alloc < len(ALLOC_TILES)
            assert ALLOC_TILES[alloc] <= info["available"][stage]


def test_heuristics_are_deterministic() -> None:
    env = _env()
    wcfg = _wcfg()
    for name, decide in HEURISTICS.items():
        a = run_scheduler(decide, env, wcfg, seed=1, max_steps=300)
        b = run_scheduler(decide, env, wcfg, seed=1, max_steps=300)
        assert a == b


def test_run_scheduler_reports_positive_jct() -> None:
    env = _env()
    wcfg = _wcfg()
    for name, decide in HEURISTICS.items():
        jct = run_scheduler(decide, env, wcfg, seed=1, max_steps=300)
        assert jct > 0.0
        assert jct < float("inf")


def test_evaluate_runs_all_deciders_over_seeds() -> None:
    env = _env()
    wcfg = _wcfg()
    results = evaluate(HEURISTICS, env, wcfg, seeds=[0, 1], max_steps=300)
    assert set(results) == set(HEURISTICS)
    for vals in results.values():
        assert len(vals) == 2
        assert all(v > 0 for v in vals)
    summary = summarize(results)
    assert set(summary) == set(results)
    for _m, s in summary.values():
        assert s >= 0.0


def test_fifo_matches_baseline_rollout() -> None:
    env = _env()
    wcfg = _wcfg()
    jct_policy = run_scheduler(HEURISTICS["fifo"], env, wcfg, seed=2, max_steps=300)
    ref = baseline_rollout(env, wcfg, seed=2, max_steps=300)
    jct_rollout = -ref["rewards"].sum().item() / wcfg.num_jobs
    assert abs(jct_policy - jct_rollout) < 1e-6


def test_policy_decide_runs_end_to_end() -> None:
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    decider = policy_decide(pol)
    jct = run_scheduler(decider, env, _wcfg(), seed=3, max_steps=300)
    assert jct > 0.0
    assert torch.isfinite(torch.tensor(jct))
