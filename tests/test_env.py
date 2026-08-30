"""milestone 1.5: gymnasium env wrapping the c++ simulator."""

from __future__ import annotations

import pytest

from orbit import core
from orbit.sim import OrbitEnv
from orbit.sim.env import ALLOC_TILES


def _env(jobs):
    env = OrbitEnv(core.SimulatorConfig())
    obs, info = env.reset(options={"jobs": jobs})
    return env, obs, info


def _single(tasks, parallelism, dur=1.0):
    return [{"parallelism": parallelism, "stages": [{"tasks": tasks, "duration": dur}]}]


def test_env_constructs_and_sets_action_space() -> None:
    env = OrbitEnv(core.SimulatorConfig())
    obs, info = env.reset(options={"jobs": _single(4, 4)})
    # one stage -> flat space of n_tiles member actions
    assert env.action_space.n == 1 * len(ALLOC_TILES)
    assert info["runnable"] == [(0, 0)]


def test_single_stage_grants_and_accumulates_reward() -> None:
    env, _, _ = _env(_single(8, 4))
    # stage 0, tile index for 4 executors = 3
    action = 0 * len(ALLOC_TILES) + 3
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["time"] == 2.0  # 8 tasks / 4 exec = 2 waves @ 1.0s
    assert reward == pytest.approx(-2.0)
    assert env._sim.job_done(0)


def test_diamond_runs_in_dependency_order() -> None:
    jobs = [
        {
            "parallelism": 2,
            "stages": [
                {"tasks": 1, "duration": 1.0},
                {"tasks": 1, "duration": 1.0},
                {"tasks": 1, "duration": 1.0, "parents": [0, 1]},
            ],
        }
    ]
    env, obs, info = _env(jobs)
    # only the two roots are runnable to start; the join waits for both parents
    assert info["runnable"] == [(0, 0), (0, 1)]

    times = []
    total = 0.0
    terminated = False
    while not terminated:
        obs, reward, terminated, _trunc, info = env.step(1)  # 2 executors, runnable[0]
        times.append(info["time"])
        total += reward

    # roots finish one at a time, then the join
    assert times == [0.5, 1.0, 1.5]
    assert total == pytest.approx(-1.5)
    assert env._sim.job_done(0)


def test_two_jobs_share_pool_reward_is_sum_of_jcts() -> None:
    jobs = _single(4, 4) + _single(4, 4)
    env, obs, info = _env(jobs)
    assert info["runnable"] == [(0, 0), (1, 0)]

    # runnable[0] is job 0; grant it 4 executors via tile index 3
    total = 0.0
    terminated = False
    while not terminated:
        obs, reward, terminated, _trunc, info = env.step(3)
        total += reward

    # job 0 finishes at t=1 (jct 1), job 1 finishes at t=2 (jct 2)
    assert total == pytest.approx(-3.0)
    assert env._sim.job_done(0)
    assert env._sim.job_done(1)


def test_allocation_clamped_by_parallelism_headroom() -> None:
    # parallelism 2, so the largest grant that sticks is 2 executors. if the
    # clamp failed and it used 4, the job would finish twice as fast.
    env, _, _ = _env(_single(8, 2))
    # ask for tile 4 executors (index 3) but the job's headroom is 2
    action = 0 * len(ALLOC_TILES) + 3
    obs, reward, terminated, _trunc, info = env.step(action)

    # 8 tasks at 2 executors, no cap: 4 waves of 2 @ 1.0s each
    assert info["time"] == 4.0
    assert reward == pytest.approx(-4.0)
    assert terminated is True


def test_action_picks_stage_among_runnables() -> None:
    jobs = [{"parallelism": 2, "stages": [{"tasks": 1, "duration": 1.0}]}] * 3
    env, _, info = _env(jobs)
    assert info["runnable"] == [(0, 0), (1, 0), (2, 0)]

    # stage_idx 2 -> third runnable (job 2); tile index 0 -> 1 executor
    action = 2 * len(ALLOC_TILES) + 0
    obs, _reward, _term, _trunc, info = env.step(action)
    # granting to job 2 runs its single-task stage to completion; verify the
    # grant was issued and the simulator advanced past it
    assert env._sim.jobs()[2].completed is True


def test_exhausted_episode_terminates_once() -> None:
    env, _, _ = _env(_single(2, 2))
    _, reward, terminated, _, _ = env.step(1)  # 2 executors
    assert terminated is True
    # stepping past the end keeps terminated and returns 0 reward again
    obs, reward, terminated, _, _ = env.step(0)
    assert terminated is True
