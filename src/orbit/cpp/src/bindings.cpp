#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "orbit/dag.hpp"
#include "orbit/executor.hpp"
#include "orbit/observation.hpp"
#include "orbit/simulator.hpp"

namespace py = pybind11;
using namespace orbit;

PYBIND11_MODULE(_orbit_core, m) {
  m.doc() = "orbit c++ core: discrete-event dag cluster simulator.";

  py::class_<Stage>(m, "Stage")
      .def(py::init<>())
      .def_readwrite("parents", &Stage::parents)
      .def_readwrite("children", &Stage::children)
      .def_readwrite("total_tasks", &Stage::total_tasks)
      .def_readwrite("tasks_remaining", &Stage::tasks_remaining)
      .def_readwrite("avg_task_duration", &Stage::avg_task_duration)
      .def_readwrite("assigned_executors", &Stage::assigned_executors)
      .def_readwrite("mu_cap", &Stage::mu_cap)
      .def_readwrite("completed", &Stage::completed)
      .def_readwrite("wave_count", &Stage::wave_count)
      .def("is_root", &Stage::is_root)
      .def("is_leaf", &Stage::is_leaf)
      .def("runnable", &Stage::runnable);

  py::class_<Job>(m, "Job")
      .def(py::init<>())
      .def_readwrite("stages", &Job::stages)
      .def_readwrite("parallelism_limit", &Job::parallelism_limit)
      .def_readwrite("assigned_executors", &Job::assigned_executors)
      .def_readwrite("starting_executors", &Job::starting_executors)
      .def_readwrite("active_executors", &Job::active_executors)
      .def_readwrite("arrival_time", &Job::arrival_time)
      .def_readwrite("active_stage", &Job::active_stage)
      .def_readwrite("completed", &Job::completed)
      .def_readwrite("completion_time", &Job::completion_time);

  py::class_<StageObs>(m, "StageObs")
      .def(py::init<>())
      .def_readonly("node_id", &StageObs::node_id)
      .def_readonly("job_index", &StageObs::job_index)
      .def_readonly("stage_index", &StageObs::stage_index)
      .def_readonly("tasks_remaining", &StageObs::tasks_remaining)
      .def_readonly("avg_task_duration", &StageObs::avg_task_duration)
      .def_readonly("assigned_executors", &StageObs::assigned_executors)
      .def_readonly("mu_cap", &StageObs::mu_cap)
      .def_readonly("parallelism_limit", &StageObs::parallelism_limit)
      .def_readonly("headroom", &StageObs::headroom)
      .def_readonly("wave_count", &StageObs::wave_count)
      .def_readonly("completed", &StageObs::completed)
      .def_readonly("is_root", &StageObs::is_root)
      .def_readonly("is_leaf", &StageObs::is_leaf)
      .def_readonly("runnable", &StageObs::runnable)
      .def_readonly("age", &StageObs::age);

  py::class_<Observation>(m, "Observation")
      .def(py::init<>())
      .def_readonly("nodes", &Observation::nodes)
      .def_readonly("runnable", &Observation::runnable)
      .def_readonly("edge_src", &Observation::edge_src)
      .def_readonly("edge_dst", &Observation::edge_dst);

  py::class_<Simulator>(m, "Simulator")
      .def(py::init<Simulator::Config>())
      .def("reset", &Simulator::reset)
      .def("add_job", &Simulator::add_job)
      .def("add_job_at", &Simulator::add_job_at)
      .def("num_idle", &Simulator::num_idle)
      .def("available_for", &Simulator::available_for)
      .def("jobs", &Simulator::jobs)
      .def("now", &Simulator::now)
      .def("add_executors", &Simulator::add_executors)
      .def("job_done", &Simulator::job_done)
      .def("observe", &Simulator::observe)
      .def("step", &Simulator::step)
      .def("run_until_idle", &Simulator::run_until_idle)
      .def("completed_since_step", &Simulator::completed_since_step)
      .def("jct_since_step", &Simulator::jct_since_step);

  py::class_<Simulator::Config>(m, "SimulatorConfig")
      .def(py::init<>())
      .def_readwrite("num_executors", &Simulator::Config::num_executors)
      .def_readwrite("seed", &Simulator::Config::seed)
      .def_readwrite("jvm_startup_delay",
                     &Simulator::Config::jvm_startup_delay)
      .def_readwrite("first_wave_slowdown",
                     &Simulator::Config::first_wave_slowdown)
      .def_readwrite("executor_rate", &Simulator::Config::executor_rate);
}
