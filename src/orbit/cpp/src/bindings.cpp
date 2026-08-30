#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "orbit/dag.hpp"
#include "orbit/executor.hpp"
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
      .def("is_root", &Stage::is_root)
      .def("is_leaf", &Stage::is_leaf)
      .def("runnable", &Stage::runnable);

  py::class_<Job>(m, "Job")
      .def(py::init<>())
      .def_readwrite("stages", &Job::stages)
      .def_readwrite("parallelism_limit", &Job::parallelism_limit)
      .def_readwrite("assigned_executors", &Job::assigned_executors)
      .def_readwrite("arrival_time", &Job::arrival_time);

  py::class_<Simulator>(m, "Simulator")
      .def(py::init<Simulator::Config>())
      .def("reset", &Simulator::reset)
      .def("num_idle", &Simulator::num_idle)
      .def("jobs", &Simulator::jobs)
      .def("now", &Simulator::now)
      .def("add_executors", &Simulator::add_executors)
      .def("run_until_idle", &Simulator::run_until_idle);

  py::class_<Simulator::Config>(m, "SimulatorConfig")
      .def(py::init<>())
      .def_readwrite("num_executors", &Simulator::Config::num_executors);
}
