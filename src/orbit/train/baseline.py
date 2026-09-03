"""a deterministic heuristic reference scheduler for a variance-reducing baseline.

the heuristic picks the first runnable stage and grants it every executor the
job allows. running it over the same seeded episode as the policy gives a
reference reward trace whose returns-to-go subtract from the policy returns.
"""

from __future__ import annotations

import torch

from ..sim import ALLOC_TILES, OrbitEnv, WorkloadConfig


def heuristic_action(info: dict) -> tuple[int, int]:
    """a fifo plus greedy grant decision from an info dict.

    returns (stage_action, alloc_action): the first runnable stage, and the
    largest tile that fits the executors available to that stage.
    """
    available = info["available"]
    if not available or available[0] <= 0:
        return (0, 0)
    tile = 0
    for i, size in enumerate(ALLOC_TILES):
        if size <= available[0]:
            tile = i
        else:
            break
    return (0, tile)


def baseline_rollout(
    env: OrbitEnv,
    wcfg: WorkloadConfig,
    seed: int | None = None,
    max_steps: int = 1000,
) -> dict:
    """run one episode with the heuristic, returning a reward trace.

    only the rewards matter here; the per decision returns-to-go are computed
    as the differential baseline inside the loss.
    """
    obs, info = env.reset(options={"workload": wcfg, "seed": seed})
    rewards: list[float] = []
    terminated = False

    for _ in range(max_steps):
        if terminated:
            break
        step_action = heuristic_action(info) if info["runnable"] else (0, 0)
        obs, reward, terminated, _truncated, info = env.step(step_action)
        rewards.append(reward)

    return {
        "rewards": torch.tensor(rewards, dtype=torch.float32),
    }


__all__ = ["baseline_rollout", "heuristic_action"]
