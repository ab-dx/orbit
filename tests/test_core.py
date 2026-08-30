"""smoke + milestone 1.1 tests for the compiled c++ core."""

from __future__ import annotations

from math import ceil

from orbit import __version__, core


def _single_stage_job(stage, parallelism):
    job = core.Job()
    job.parallelism_limit = parallelism
    job.stages = [stage]
    return job


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_core_importable() -> None:
    # the c++ extension must expose the simulator and its data structures.
    assert hasattr(core, "Simulator")
    assert hasattr(core, "SimulatorConfig")
    assert hasattr(core, "Job")
    assert hasattr(core, "Stage")


def test_simulator_basic_roundtrip() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 50
    sim = core.Simulator(cfg)

    # all executors start idle
    assert sim.num_idle() == 50
    assert len(sim.jobs()) == 0

    sim.reset()
    assert sim.num_idle() == 50
    assert sim.now() == 0.0


def test_stage_flags() -> None:
    st = core.Stage()
    st.total_tasks = 4
    st.tasks_remaining = 4
    st.avg_task_duration = 2.5

    assert st.runnable() is True
    assert st.is_leaf() is True  # no children yet
    assert st.is_root() is True  # no parents yet
    assert st.completed is False


def test_job_carries_stages() -> None:
    job = core.Job()
    stage = core.Stage()
    stage.total_tasks = 8
    stage.tasks_remaining = 8

    job.stages = [stage]
    job.parallelism_limit = 4

    assert job.stages[0].tasks_remaining == 8
    assert job.parallelism_limit == 4


def test_single_stage_parallel_run() -> None:
    # one stage, 8 tasks, 4 executors, 2.5s each.
    # 8 / 4 = 2 waves, so it should take exactly 2 * 2.5 = 5.0s.
    st = core.Stage()
    st.total_tasks = 8
    st.tasks_remaining = 8
    st.avg_task_duration = 2.5

    cfg = core.SimulatorConfig()
    cfg.num_executors = 50
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)
    sim.run_until_idle()

    assert sim.job_done(idx)
    completed = sim.jobs()[idx]
    assert completed.stages[0].completed is True
    assert completed.stages[0].tasks_remaining == 0
    assert completed.completion_time == 5.0


def test_single_stage_partial_parallelism() -> None:
    # 10 tasks but only 3 executors -> ceil(10/3) = 4 waves, 2.0s each.
    st = core.Stage()
    st.total_tasks = 10
    st.tasks_remaining = 10
    st.avg_task_duration = 2.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=3))
    sim.add_executors(idx, 3)
    sim.run_until_idle()

    assert sim.job_done(idx)
    assert sim.jobs()[idx].completion_time == ceil(10 / 3) * 2.0


def test_chain_releases_dependents() -> None:
    # two sequential stages; stage 1 only runs after stage 0 completes.
    s0 = core.Stage()
    s0.total_tasks = 2
    s0.tasks_remaining = 2
    s0.avg_task_duration = 1.0

    s1 = core.Stage()
    s1.parents = [0]
    s1.total_tasks = 2
    s1.tasks_remaining = 2
    s1.avg_task_duration = 1.0

    job = core.Job()
    job.parallelism_limit = 2
    job.stages = [s0, s1]

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)

    idx = sim.add_job(job)
    sim.add_executors(idx, 2)
    sim.run_until_idle()

    done = sim.jobs()[idx]
    assert done.completed is True
    assert done.completion_time == 2.0  # 1 wave each of 2 tasks at 1.0s
    assert done.stages[1].completed is True


def test_join_waits_for_all_parents() -> None:
    # two independent roots feeding one join stage.
    s0 = core.Stage()
    s0.total_tasks = 1
    s0.tasks_remaining = 1
    s0.avg_task_duration = 1.0

    s1 = core.Stage()
    s1.total_tasks = 1
    s1.tasks_remaining = 1
    s1.avg_task_duration = 1.0

    s2 = core.Stage()
    s2.parents = [0, 1]
    s2.total_tasks = 1
    s2.tasks_remaining = 1
    s2.avg_task_duration = 1.0

    job = core.Job()
    job.parallelism_limit = 1  # only 1 executor: stages run one at a time
    job.stages = [s0, s1, s2]

    cfg = core.SimulatorConfig()
    cfg.num_executors = 5
    sim = core.Simulator(cfg)

    idx = sim.add_job(job)
    sim.add_executors(idx, 1)
    sim.run_until_idle()

    done = sim.jobs()[idx]
    # s0 at 1.0, s1 at 2.0, then join runs after both complete -> 3.0
    assert done.completed is True
    assert done.completion_time == 3.0
    assert done.stages[2].completed is True


def test_add_executors_respects_parallelism_limit() -> None:
    st = core.Stage()
    st.total_tasks = 100
    st.tasks_remaining = 100
    st.avg_task_duration = 1.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 10)  # ask for 10, but limit is 4
    sim.run_until_idle()

    done = sim.jobs()[idx]
    assert done.assigned_executors == 4
    # 100 tasks / 4 executors = 25 waves @ 1.0s
    assert done.completion_time == 25.0
