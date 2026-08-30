"""smoke tests for the compiled c++ core and package wiring (phase 0)."""

from __future__ import annotations

from orbit import __version__, core


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

    assert sim.num_idle() == 0  # no executors adopted yet (Phase 1)
    assert len(sim.jobs()) == 0

    sim.reset()
    assert sim.now() == 0.0


def test_stage_flags() -> None:
    st = core.Stage()
    st.total_tasks = 4
    st.tasks_remaining = 4
    st.avg_task_duration = 2.5

    assert st.runnable() is True
    assert st.is_leaf() is True  # no children yet
    assert st.is_root() is True  # no parents yet


def test_job_carries_stages() -> None:
    job = core.Job()
    stage = core.Stage()
    stage.total_tasks = 8
    stage.tasks_remaining = 8

    job.stages = [stage]
    job.parallelism_limit = 4

    assert job.stages[0].tasks_remaining == 8
    assert job.parallelism_limit == 4
