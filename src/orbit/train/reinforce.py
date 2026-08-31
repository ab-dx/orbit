"""reinforce training for the scheduling policy.

rolls out episodes, records log probs and negative jct rewards, then steps the
policy along the gradient. a running-average return baseline cuts variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..agent import Policy, graph_input
from ..sim import OrbitEnv, WorkloadConfig


def rollout(
    policy: Policy,
    env: OrbitEnv,
    wcfg: WorkloadConfig,
    seed: int | None = None,
    max_steps: int = 1000,
) -> dict:
    """run one episode, collecting log probs and rewards.

    only steps with a runnable stage make a decision and get a log prob;
    waiting steps add none.

    returns the stacked decision log probs, per step rewards, total reward
    (negative total jct) and decision count.
    """
    obs, info = env.reset(options={"workload": wcfg, "seed": seed})

    h = None
    log_probs: list[torch.Tensor] = []
    decision_ix: list[int] = []
    rewards: list[float] = []
    total_reward = 0.0
    terminated = False

    for _ in range(max_steps):
        if terminated:
            break

        if info["runnable"]:
            # encode on demand; skip empty pre arrival dags
            if h is None:
                with torch.no_grad():
                    h = policy.encode(*graph_input(obs))
            runnable_ids = info["runnable_node_ids"]
            available = info["available"]
            action = policy.sample(h, runnable_ids, available)
            lp = policy.log_prob(h, runnable_ids, available, action)
            log_probs.append(lp)
            decision_ix.append(len(rewards))
            step_action = (action.stage, action.alloc)
        else:
            # nothing schedulable; no-op and let arrivals drive time
            step_action = (0, 0)

        obs, reward, terminated, _truncated, info = env.step(step_action)
        rewards.append(reward)
        total_reward += reward
        # graph changed; re-encode next round
        h = None

    return {
        "log_probs": torch.stack(log_probs) if log_probs else torch.tensor([]),
        "decision_ix": torch.tensor(decision_ix, dtype=torch.long),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "total_reward": total_reward,
        "decisions": len(decision_ix),
    }


def returns_to_go(rewards: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """discounted sum of future rewards per step, backwards."""
    rets = torch.empty_like(rewards)
    acc = 0.0
    for t in range(rewards.numel() - 1, -1, -1):
        acc = rewards[t] + gamma * acc
        rets[t] = acc
    return rets


def reinforce_loss(traj: dict, baseline: float = 0.0, gamma: float = 1.0) -> torch.Tensor:
    """the reinforce objective: log probs weighted by returns minus baseline.

    scalar and differentiable; the mean keeps magnitude stable across episodes
    with different decision counts.
    """
    lps = traj["log_probs"]
    if lps.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    # weight each decision by the return at its own step
    rets = returns_to_go(traj["rewards"], gamma=gamma)[traj["decision_ix"]]
    advantages = rets - baseline
    return -(advantages * lps).mean()


@dataclass
class TrainConfig:
    """tunables for one reinforce training run."""

    iters: int = 100
    gamma: float = 1.0            # discount on returns
    lr: float = 3e-3              # lr for a fresh adam optimizer
    baseline_window: int = 20     # window for the return baseline
    seed: int | None = None
    max_steps: int = 1000


def train(
    policy: Policy,
    env: OrbitEnv,
    wcfg: WorkloadConfig,
    cfg: TrainConfig = TrainConfig(),
    progress: Callable[[int, float, float], None] | None = None,
) -> list[float]:
    """run a reinforce loop, returning the total reward per iteration.

    each iteration rolls out once, computes the loss against a running-average
    baseline and takes one optimizer step. the policy is updated in place.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    returns_log: list[float] = []
    window: list[float] = []

    for it in range(cfg.iters):
        traj = rollout(
            policy,
            env,
            wcfg,
            seed=None if cfg.seed is None else cfg.seed + it,
            max_steps=cfg.max_steps,
        )
        iteration_return = traj["total_reward"]
        returns_log.append(iteration_return)

        # baseline from past returns only, so it stays unbiased
        baseline = sum(window) / len(window) if window else 0.0
        window.append(iteration_return)
        if len(window) > cfg.baseline_window:
            window.pop(0)

        optimizer.zero_grad()
        loss = reinforce_loss(traj, baseline=baseline, gamma=cfg.gamma)
        loss.backward()
        optimizer.step()

        if progress is not None:
            progress(it, iteration_return, baseline)

    return returns_log


__all__ = [
    "TrainConfig",
    "reinforce_loss",
    "returns_to_go",
    "rollout",
    "train",
]
