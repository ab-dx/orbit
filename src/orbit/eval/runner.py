"""run schedulers over seeded episodes and compare average jct."""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean, stdev

import torch

from ..agent import Policy, graph_input

# a schedulers decision: (stage_action, alloc_action) from the current view.
Decider = Callable[[object, dict], tuple[int, int]]


def policy_decide(policy: Policy) -> Decider:
    """wrap a trained policy as a decide(obs, info) callable."""

    def decide(obs, info: dict) -> tuple[int, int]:
        with torch.no_grad():
            h = policy.encode(*graph_input(obs))
        action = policy.sample(h, info["runnable_node_ids"], info["available"])
        return (action.stage, action.alloc)

    return decide


def run_scheduler(
    decide: Decider,
    env,
    wcfg,
    seed: int | None = None,
    max_steps: int = 1000,
) -> float:
    """average job completion time for one seeded episode under decide."""
    obs, info = env.reset(options={"workload": wcfg, "seed": seed})
    total_reward = 0.0
    terminated = False
    for _ in range(max_steps):
        if terminated:
            break
        step_action = decide(obs, info) if info["runnable"] else (0, 0)
        obs, reward, terminated, _truncated, info = env.step(step_action)
        total_reward += reward
    return -total_reward / max(env._n_jobs, 1)


def evaluate(
    deciders: dict[str, Decider],
    env,
    wcfg,
    seeds: list[int],
    max_steps: int = 1000,
) -> dict[str, list[float]]:
    """average jct per decider across every seed."""
    results: dict[str, list[float]] = {}
    for name, decide in deciders.items():
        results[name] = [
            run_scheduler(decide, env, wcfg, seed=s, max_steps=max_steps)
            for s in seeds
        ]
    return results


def summarize(results: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    """mean and std of each decider's average jct."""
    return {
        name: (mean(vals), stdev(vals) if len(vals) > 1 else 0.0)
        for name, vals in results.items()
    }


__all__ = ["Decider", "evaluate", "policy_decide", "run_scheduler", "summarize"]
