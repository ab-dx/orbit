"""train a scheduling policy, compare it to classic heuristics, save charts.

this reproduces the results section of the readme:
it trains a fresh policy with reinforce (differential rewards on), evaluates
it against fifo/sjf/fair/weighted_fair on the same workloads, and writes the
training curve and evaluation charts into assets/.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from orbit import core
from orbit.agent import Policy, save_policy
from orbit.eval import HEURISTICS, evaluate, policy_decide, summarize
from orbit.sim import OrbitEnv, WorkloadConfig
from orbit.train import TrainConfig, train


def default_workload() -> WorkloadConfig:
    """a moderate mixed-topology workload, small enough to run quickly."""
    return WorkloadConfig(
        job_arrival_rate=0.8,
        num_jobs=8,
        cluster_size=16,
        max_stages=5,
        min_tasks_per_stage=2,
        max_tasks_per_stage=8,
        min_avg_task_duration=0.5,
        max_avg_task_duration=2.0,
        topology="mixed",
    )


def plot_training(returns: list[float], num_jobs: int, path: str) -> None:
    """plot per-iteration average jct (negative reward per episode)."""
    jct = [-r / num_jobs for r in returns]
    iters = np.arange(1, len(jct) + 1)
    window = max(5, len(jct) // 10)
    kernel = np.ones(window) / window
    smooth = np.convolve(jct, kernel, mode="valid")
    x_smooth = np.arange(window, len(jct) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, jct, alpha=0.25, linewidth=0.8, color="#4c72b0")
    ax.plot(x_smooth, smooth, linewidth=2.0, color="#4c72b0")
    ax.set_xlabel("training iteration")
    ax.set_ylabel("average jct per episode (seconds)")
    ax.set_title("reinforce training curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eval(summary: dict[str, tuple[float, float]], path: str) -> None:
    """horizontal bars of mean jct with std error bars, best first."""
    names = sorted(summary, key=lambda n: summary[n][0])
    means = [summary[n][0] for n in names]
    stds = [summary[n][1] for n in names]
    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(y, means, xerr=stds, height=0.6, color="#c44e52", capsize=4)
    ax.set_yticks(y, labels=names)
    ax.set_xlabel("average jct (seconds)")
    ax.set_title("learned policy vs classic heuristics")
    for i, m in enumerate(means):
        ax.text(m, i, f" {m:.1f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="train + evaluate the orbit scheduling policy"
    )
    parser.add_argument("--iters", type=int, default=1200,
                        help="training iterations (default matches the readme run, ~6 min)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--checkpoint", default="policy.pt")
    parser.add_argument("--assets", default="assets")
    args = parser.parse_args()

    env = OrbitEnv(core.SimulatorConfig())
    wcfg = default_workload()

    torch.manual_seed(args.train_seed)
    policy = Policy(in_dim=12, hidden_dim=64)
    cfg = TrainConfig(
        iters=args.iters,
        seed=args.train_seed,
        max_steps=args.max_steps,
        differential=True,
    )
    print(f"training {args.iters} iterations with differential rewards...")
    returns = train(
        policy,
        env,
        wcfg,
        cfg,
        progress=lambda it, ret, base: print(
            f"  iter {it + 1}/{args.iters}  avg_jct={-ret / wcfg.num_jobs:.2f}s",
            end="\r",
        ),
    )
    print("\ntraining done, saving policy...")
    save_policy(policy, args.checkpoint)

    decile = max(1, len(returns) // 10)
    first = np.mean([-r / wcfg.num_jobs for r in returns[:decile]])
    last = np.mean([-r / wcfg.num_jobs for r in returns[-decile:]])
    print(
        f"policy improved from {first:.2f}s to {last:.2f}s "
        f"avg jct over training ({100 * (1 - last / first):.0f}% reduction)"
    )

    print("evaluating against heuristics...")
    deciders: dict = {"policy": policy_decide(policy)}
    deciders.update(HEURISTICS)
    results = evaluate(deciders, env, wcfg, seeds=args.seeds, max_steps=args.max_steps)
    summary = summarize(results)

    os.makedirs(args.assets, exist_ok=True)
    curve = os.path.join(args.assets, "training_curve.png")
    bars = os.path.join(args.assets, "eval_results.png")
    plot_training(returns, wcfg.num_jobs, curve)
    plot_eval(summary, bars)
    print(f"charts written to {curve} and {bars}")

    print("\naverage jct (seconds) per decider:")
    for name in summary:
        mean, std = summary[name]
        print(f"  {name:>12}  {mean:7.2f} +/- {std:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
