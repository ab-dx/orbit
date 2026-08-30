"""smoke + milestone 1.1 tests for the compiled c++ core."""

from __future__ import annotations

import numpy as np
import pytest

from orbit import __version__, core
from orbit.sim.observe import edge_index, node_features, runnable_set


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
    # 10 tasks but only 3 executors. rate = 3 * (1/2.0) = 1.5 tasks/sec.
    # three full waves of 3 (2.0s each) + a final partial wave of 1 task
    # (1/1.5 = 0.6667s) = 6.6667s. the piecewise rate model speeds up the
    # trailing partial wave instead of charging a full duration for it.
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
    assert sim.jobs()[idx].completion_time == pytest.approx(6.0 + 2.0 / 3.0)


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
    assert sim.jobs()[idx].assigned_executors == 4  # capped by limit
    sim.run_until_idle()

    done = sim.jobs()[idx]
    assert done.assigned_executors == 0  # released back to the pool
    # 100 tasks / 4 executors = 25 waves @ 1.0s
    assert done.completion_time == 25.0
    assert sim.num_idle() == 10


# ---- milestone 1.2 ----


def test_jvm_startup_delay_defers_work() -> None:
    # granted executors only work after the startup delay.
    st = core.Stage()
    st.total_tasks = 4
    st.tasks_remaining = 4
    st.avg_task_duration = 1.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    cfg.jvm_startup_delay = 2.0
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)

    # before the loop runs, executors are starting, not active
    job = sim.jobs()[idx]
    assert job.starting_executors == 4
    assert job.active_executors == 0
    assert sim.num_idle() == 6  # 4 taken out of idle immediately

    sim.run_until_idle()
    assert sim.job_done(idx)
    # 2.0s startup + 1 wave of 4 tasks @ 1.0s
    assert sim.jobs()[idx].completion_time == 3.0


def test_first_wave_slowdown_only_first_wave() -> None:
    # 8 tasks / 4 exec = 2 waves; wave 1 slowed by 2.0, wave 2 normal.
    st = core.Stage()
    st.total_tasks = 8
    st.tasks_remaining = 8
    st.avg_task_duration = 1.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    cfg.first_wave_slowdown = 2.0
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)
    sim.run_until_idle()

    assert sim.job_done(idx)
    assert sim.jobs()[idx].completion_time == 3.0  # 2.0 + 1.0
    assert sim.jobs()[idx].stages[0].wave_count == 2


def test_executors_released_back_to_idle_on_completion() -> None:
    st = core.Stage()
    st.total_tasks = 10
    st.tasks_remaining = 10
    st.avg_task_duration = 1.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 8
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)
    assert sim.num_idle() == 4

    sim.run_until_idle()
    # the job's 4 executors are returned to the pool
    assert sim.num_idle() == 8
    assert sim.jobs()[idx].assigned_executors == 0


def test_two_jobs_run_concurrently() -> None:
    # two independent jobs share the pool and both progress.
    def one(tasks):
        st = core.Stage()
        st.total_tasks = tasks
        st.tasks_remaining = tasks
        st.avg_task_duration = 1.0
        return _single_stage_job(st, parallelism=2)

    cfg = core.SimulatorConfig()
    cfg.num_executors = 4
    sim = core.Simulator(cfg)

    a = sim.add_job(one(4))
    b = sim.add_job(one(4))
    sim.add_executors(a, 2)
    sim.add_executors(b, 2)

    # 2 exec slots still free
    assert sim.num_idle() == 0  # 4 granted total

    sim.run_until_idle()
    assert sim.job_done(a)
    assert sim.job_done(b)
    # each job: 4 tasks / 2 exec = 2 waves @ 1.0s
    assert sim.jobs()[a].completion_time == 2.0
    assert sim.jobs()[b].completion_time == 2.0
    assert sim.num_idle() == 4


def test_executor_migration_between_jobs_with_delay() -> None:
    # job A uses 4 executors and finishes, freeing them; job B then takes
    # them with a startup delay.
    def one(tasks):
        st = core.Stage()
        st.total_tasks = tasks
        st.tasks_remaining = tasks
        st.avg_task_duration = 1.0
        return _single_stage_job(st, parallelism=4)

    cfg = core.SimulatorConfig()
    cfg.num_executors = 4
    cfg.jvm_startup_delay = 1.0
    sim = core.Simulator(cfg)

    a = sim.add_job(one(4))
    sim.add_executors(a, 4)
    # run until A completes but B has not been launched yet
    sim.run_until_idle()
    assert sim.job_done(a)
    assert sim.num_idle() == 4

    b = sim.add_job(one(4))
    sim.add_executors(b, 4)
    assert sim.num_idle() == 0
    sim.run_until_idle()

    assert sim.job_done(b)
    # B starts at t=2 (after A finished); 1.0s startup + 1.0s run = t=4
    assert sim.jobs()[b].completion_time == 4.0
    assert sim.num_idle() == 4


# ---- milestone 1.3 ----


def test_high_parallelism_slower_than_linear() -> None:
    # 100 tasks / 4 exec with avg 1.0 would be 25s at linear speedup. the
    # stage's mu_cap of 2.0 tasks/sec saturates throughput: rate = min(4,2)=2,
    # so each full wave of 4 tasks takes 2.0s -> 25 * 2.0 = 50s.
    st = core.Stage()
    st.total_tasks = 100
    st.tasks_remaining = 100
    st.avg_task_duration = 1.0
    st.mu_cap = 2.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)
    sim.run_until_idle()

    assert sim.job_done(idx)
    assert sim.jobs()[idx].completion_time == 50.0


def test_high_parallelism_saturates_without_cap() -> None:
    # same workload but no mu_cap: linear speedup holds at 25s.
    st = core.Stage()
    st.total_tasks = 100
    st.tasks_remaining = 100
    st.avg_task_duration = 1.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)
    sim.run_until_idle()

    assert sim.jobs()[idx].completion_time == 25.0


def test_step_advances_decision_points() -> None:
    # 8 tasks / 4 exec -> two waves of 1.0s. each step processes exactly the
    # next scheduled event: startup, wave1, wave2.
    st = core.Stage()
    st.total_tasks = 8
    st.tasks_remaining = 8
    st.avg_task_duration = 1.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)

    # step 1 runs the startup event at t=0 (no-op clock advance)
    assert sim.step() is True
    assert sim.now() == 0.0
    assert sim.completed_since_step() == 0

    # step 2 runs wave1, done at t=1.0
    assert sim.step() is True
    assert sim.now() == 1.0
    assert sim.completed_since_step() == 0

    # step 3 runs wave2, done at t=2.0 -> job completes
    assert sim.step() is True
    assert sim.now() == 2.0
    assert sim.job_done(idx)
    assert sim.completed_since_step() == 1
    assert sim.jct_since_step() == 2.0

    # queue empty; stepping again returns false
    assert sim.step() is False


def test_step_runs_until_idle_completes_job() -> None:
    st = core.Stage()
    st.total_tasks = 4
    st.tasks_remaining = 4
    st.avg_task_duration = 1.0

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)

    idx = sim.add_job(_single_stage_job(st, parallelism=4))
    sim.add_executors(idx, 4)

    while sim.step():
        pass

    assert sim.job_done(idx)
    assert sim.jobs()[idx].completion_time == 1.0


def test_jct_accumulates_per_step() -> None:
    # two identical single-wave jobs complete in separate steps; each step
    # reports exactly the jct of the job finishing in that interval.
    def one(tasks):
        st = core.Stage()
        st.total_tasks = tasks
        st.tasks_remaining = tasks
        st.avg_task_duration = 1.0
        return _single_stage_job(st, parallelism=4)

    cfg = core.SimulatorConfig()
    cfg.num_executors = 8
    sim = core.Simulator(cfg)

    a = sim.add_job(one(4))
    b = sim.add_job(one(4))
    sim.add_executors(a, 4)
    sim.add_executors(b, 4)

    total_jct = 0.0
    total_completed = 0
    while sim.step():
        assert sim.completed_since_step() in (0, 1)
        total_jct += sim.jct_since_step()
        total_completed += sim.completed_since_step()

    assert total_completed == 2
    # each job finishes at t=1.0 with arrival 0
    assert total_jct == pytest.approx(2.0)
    assert sim.job_done(a)
    assert sim.job_done(b)


# ---- milestone 1.4 ----


def _stage(tasks, parents=None, dur=1.0, mucap=0.0):
    st = core.Stage()
    st.total_tasks = tasks
    st.tasks_remaining = tasks
    st.avg_task_duration = dur
    st.mu_cap = mucap
    if parents:
        st.parents = parents
    return st


def _diamond_job(limit=4):
    s0 = _stage(5, dur=2.0)
    s1 = _stage(5, dur=3.0)
    s2 = _stage(8, parents=[0, 1], mucap=2.0)
    job = core.Job()
    job.parallelism_limit = limit
    job.stages = [s0, s1, s2]
    return job


def test_observe_covers_every_node() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    obs = sim.observe()
    # 3 nodes, one per stage, globally ids 0..2
    assert len(obs.nodes) == 3
    assert [n.node_id for n in obs.nodes] == [0, 1, 2]


def test_observe_per_node_features() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    obs = sim.observe()
    s0, s1, s2 = obs.nodes
    assert (s0.job_index, s0.stage_index) == (0, 0)
    assert (s1.job_index, s1.stage_index) == (0, 1)
    assert (s2.job_index, s2.stage_index) == (0, 2)

    assert s0.tasks_remaining == 5
    assert s0.avg_task_duration == 2.0
    assert s1.avg_task_duration == 3.0
    assert s0.assigned_executors == 0
    assert s2.mu_cap == 2.0
    assert s0.parallelism_limit == 4
    # no executors granted yet, so headroom equals the whole limit
    assert s0.headroom == 4
    assert s0.wave_count == 0
    assert s0.completed == 0
    assert s0.is_root == 1
    assert s0.is_leaf == 0
    assert s2.is_leaf == 1
    assert s0.age == 0.0


def test_observe_runnable_set_excludes_blocked_join() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    obs = sim.observe()
    # roots s0 and s1 are runnable; the join s2 waits for both parents
    assert obs.runnable == [(0, 0), (0, 1)]
    assert [n.runnable for n in obs.nodes] == [1, 1, 0]


def test_observe_runnable_set_advances_with_execution() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    idx = sim.add_job(_diamond_job())
    sim.add_executors(idx, 4)
    sim.run_until_idle()

    obs = sim.observe()
    assert obs.runnable == []
    assert [n.completed for n in obs.nodes] == [1, 1, 1]
    assert sim.job_done(idx)


def test_observe_dag_edges_from_parents() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    # s0 -> s2 and s1 -> s2, in stage order
    obs = sim.observe()
    assert obs.edge_src == [0, 1]
    assert obs.edge_dst == [2, 2]


def test_observe_global_node_ids_span_jobs() -> None:
    j0 = core.Job()
    j0.parallelism_limit = 3
    j0.stages = [_stage(4), _stage(4, parents=[0])]
    j1 = core.Job()
    j1.parallelism_limit = 2
    j1.stages = [_stage(6)]

    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(j0)
    sim.add_job(j1)

    obs = sim.observe()
    assert [n.node_id for n in obs.nodes] == [0, 1, 2]
    assert [(n.job_index, n.stage_index) for n in obs.nodes] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert obs.runnable == [(0, 0), (1, 0)]
    # the single edge lives inside job0: node 0 -> node 1
    assert obs.edge_src == [0]
    assert obs.edge_dst == [1]


def test_observe_is_non_mutating() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    first = sim.observe()
    second = sim.observe()
    snapped = [
        (n.tasks_remaining, n.assigned_executors, n.runnable, n.wave_count)
        for n in first.nodes
    ]
    resnap = [
        (n.tasks_remaining, n.assigned_executors, n.runnable, n.wave_count)
        for n in second.nodes
    ]
    assert snapped == resnap
    assert sim.num_idle() == 10  # no executors granted by observing


def test_node_features_matrix_shape() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    feat = node_features(sim.observe())
    # one row per stage node, and one column per StageObs feature
    assert feat.shape == (3, 12)
    assert feat.dtype == np.float32
    # row for s2 (the join) has its known non-identity values
    assert feat[2, 0] == 8.0  # tasks_remaining
    assert feat[2, 3] == 2.0  # mu_cap
    assert feat[2, 10] == 0.0  # runnable flag while blocked
    assert feat[0, 7] == 0.0  # completed
    assert feat[0, 8] == 1.0  # is_root


def test_edge_index_matrix_shape() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    idx = edge_index(sim.observe())
    assert idx.shape == (2, 2)  # (src, dst) rows, two edges
    assert idx.dtype == np.int64
    assert idx.tolist() == [[0, 1], [2, 2]]


def test_runnable_set_helper_returns_job_stage_pairs() -> None:
    cfg = core.SimulatorConfig()
    cfg.num_executors = 10
    sim = core.Simulator(cfg)
    sim.add_job(_diamond_job())

    assert runnable_set(sim.observe()) == [(0, 0), (0, 1)]

