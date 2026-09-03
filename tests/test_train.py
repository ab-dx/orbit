from __future__ import annotations

import torch

from orbit import core
from orbit.agent import Policy
from orbit.sim import OrbitEnv, WorkloadConfig
from orbit.train import (
    Curriculum,
    TrainConfig,
    baseline_rollout,
    heuristic_action,
    reinforce_loss,
    returns_to_go,
    rollout,
    train,
    wcfg_for,
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


def test_wcfg_for_disabled_returns_target() -> None:
    target = _wcfg(num_jobs=8)
    out = wcfg_for(Curriculum(enabled=False), 0, target)
    assert out is target
    assert out.num_jobs == 8


def test_wcfg_for_ramps_job_count() -> None:
    target = _wcfg(num_jobs=10)
    cur = Curriculum(start_jobs=2, ramp_iters=4, enabled=True)

    counts = [wcfg_for(cur, it, target).num_jobs for it in range(5)]
    # start small, grow, then saturate at the target
    assert counts[0] == 2
    assert counts[-1] == 10
    assert counts == sorted(counts)


def test_wcfg_for_clamps_on_target() -> None:
    target = _wcfg(num_jobs=4)
    cur = Curriculum(start_jobs=2, ramp_iters=1000, enabled=True)
    # never overshoots the target even with a long ramp
    for it in range(0, 20, 3):
        assert wcfg_for(cur, it, target).num_jobs <= target.num_jobs


def test_train_with_curriculum_grows_episodes() -> None:
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    target = _wcfg(num_jobs=6)
    seen: list[int] = []

    def progress(it, ret, base) -> None:
        seen.append(it)

    train(
        pol,
        env,
        target,
        TrainConfig(
            iters=5,
            seed=9,
            max_steps=300,
            curriculum=Curriculum(start_jobs=2, ramp_iters=4, enabled=True),
        ),
        progress=progress,
    )

    assert len(seen) == 5
    # the final iteration ramps up to the full target workload
    assert env._n_jobs == target.num_jobs


def test_heuristic_action_uses_fifo_and_greedy_grant() -> None:
    info = {"available": [16, 4, 32]}
    stage, alloc = heuristic_action(info)
    assert stage == 0
    # largest tile (32) exceeds available 16, so 16 is chosen
    assert alloc == 4  # ALLOC_TILES[4] == 16
    assert 1 << alloc == 16


def test_heuristic_action_noop_when_nothing_available() -> None:
    stage, alloc = heuristic_action({"available": [0, 8]})
    assert (stage, alloc) == (0, 0)


def test_baseline_rollout_is_deterministic() -> None:
    env = _env()
    r1 = baseline_rollout(env, _wcfg(), seed=1, max_steps=300)
    r2 = baseline_rollout(env, _wcfg(), seed=1, max_steps=300)
    assert r1["rewards"].numel() > 0
    assert torch.equal(r1["rewards"], r2["rewards"])


def test_baseline_rollout_reproduces_policy_trace_length() -> None:
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    wcfg = _wcfg()
    traj = rollout(pol, env, wcfg, seed=2, max_steps=300)
    ref = baseline_rollout(env, wcfg, seed=2, max_steps=300)
    # same seed and action count means aligned reward traces
    assert ref["rewards"].numel() == traj["rewards"].numel()


def test_differential_loss_is_zero_for_matching_reference() -> None:
    rewards = torch.tensor([-1.0, -3.0, -2.0])
    traj = {
        "log_probs": torch.tensor([-0.5, -0.7]).detach().requires_grad_(True),
        "decision_ix": torch.tensor([0, 2], dtype=torch.long),
        "rewards": rewards,
    }
    rets = returns_to_go(rewards)
    reference = rets[traj["decision_ix"]]
    loss = reinforce_loss(traj, reference=reference)
    # policy returns equal the reference, so the advantage is zero
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_differential_loss_math() -> None:
    rewards = torch.tensor([-1.0, -3.0])
    traj = {
        "log_probs": torch.tensor([-0.5, -0.7]).detach().requires_grad_(True),
        "decision_ix": torch.tensor([0, 1], dtype=torch.long),
        "rewards": rewards,
    }
    reference = torch.tensor([-2.0, -5.0])
    rets = returns_to_go(rewards)  # [-4.0, -3.0]
    expected = -(((rets - reference) * traj["log_probs"]).mean())
    loss = reinforce_loss(traj, reference=reference)
    assert torch.allclose(loss, expected)


def test_train_with_differential_runs_and_updates() -> None:
    env = _env()
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    before = [p.detach().clone() for p in pol.parameters()]
    train(
        pol,
        env,
        _wcfg(),
        TrainConfig(iters=3, seed=5, max_steps=300, differential=True),
    )
    changed = any(
        not torch.equal(a, b) for a, b in zip(before, pol.parameters())
    )
    assert changed
