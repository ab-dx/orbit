from __future__ import annotations

import torch

from orbit import core
from orbit.agent import Policy
from orbit.sim import OrbitEnv, WorkloadConfig
from orbit.train import (
    TrainConfig,
    reinforce_loss,
    returns_to_go,
    rollout,
    train,
)

N_FEATURES = 12


def _env(wcfg: WorkloadConfig | None = None) -> OrbitEnv:
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


def test_returns_to_go_undiscounted() -> None:
    rets = returns_to_go(torch.tensor([1.0, 2.0, 3.0]), gamma=1.0)
    assert rets.tolist() == [6.0, 5.0, 3.0]


def test_returns_to_go_discounted() -> None:
    rets = returns_to_go(torch.tensor([1.0, 2.0, 3.0]), gamma=0.5)
    assert torch.allclose(
        rets, torch.tensor([1.0 + 0.5 * 2.0 + 0.25 * 3.0, 2.0 + 0.5 * 3.0, 3.0])
    )


def test_rollout_produces_valid_trajectory() -> None:
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    traj = rollout(pol, env, _wcfg(), seed=1, max_steps=300)

    assert traj["decisions"] > 0
    assert traj["log_probs"].numel() == traj["decisions"]
    assert traj["rewards"].numel() > 0
    assert torch.isfinite(traj["log_probs"]).all()
    # all jct accumulates as negative reward
    assert traj["total_reward"] < 0.0
    # decision indices stay in range of the reward trace
    assert int(traj["decision_ix"].max()) < traj["rewards"].numel()


def test_reinforce_loss_is_differentiable() -> None:
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    traj = rollout(pol, env, _wcfg(), seed=2, max_steps=300)

    loss = reinforce_loss(traj, baseline=traj["total_reward"])
    assert loss.requires_grad
    assert torch.isfinite(loss).item()

    loss.backward()
    grads = [p.grad for p in pol.parameters() if p.grad is not None]
    assert grads, "expected at least one non-none gradient"
    assert any(torch.isfinite(g).any().item() for g in grads)
    assert any((g != 0).any().item() for g in grads)


def test_reinforce_loss_empty_trajectory_is_zero() -> None:
    traj = {
        "log_probs": torch.tensor([]),
        "decision_ix": torch.tensor([], dtype=torch.long),
        "rewards": torch.tensor([], dtype=torch.float32),
    }
    loss = reinforce_loss(traj, baseline=0.0)
    assert loss.item() == 0.0


def test_train_updates_policy_parameters() -> None:
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    before = [p.detach().clone() for p in pol.parameters()]

    returns = train(
        pol,
        env,
        _wcfg(),
        TrainConfig(iters=3, seed=5, max_steps=300, lr=1e-2),
    )

    assert len(returns) == 3
    after = [p.detach().clone() for p in pol.parameters()]
    changed = any(not torch.equal(a, b) for a, b in zip(before, after))
    assert changed, "policy parameters should move under training"


def test_train_uses_prior_episodes_as_baseline() -> None:
    # after the first episode, the baseline is a nonzero prior return, so the
    # closure feed to reinforce_loss references a proper baseline value.
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    baselines: list[float] = []

    def progress(it, ret, base) -> None:
        baselines.append(base)

    train(
        pol,
        env,
        _wcfg(),
        TrainConfig(iters=4, seed=6, max_steps=300),
        progress=progress,
    )

    # the first baseline is 0 (no history); later ones reflect prior returns
    assert baselines[0] == 0.0
    assert any(b < 0.0 for b in baselines[1:])
