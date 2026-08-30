#pragma once

#include "orbit/dag.hpp"
#include "orbit/event_loop.hpp"
#include "orbit/executor.hpp"

#include <vector>

namespace orbit {

// the cluster simulator: an eventloop model of how job dags execute on a pool
// of executors. this is the rl training environment.
class Simulator {
 public:
  struct Config {
    int num_executors = 50;
  };

  explicit Simulator(Config cfg);

  // reset to an empty cluster.
  void reset();

  // number of idle executors currently available.
  int num_idle() const { return pool_.num_idle(); }

  // jobs currently in the cluster.
  const std::vector<Job>& jobs() const { return jobs_; }

  EventLoop::Clock now() const { return events_.now(); }

  // assign up to count executors to job job_index.
  void add_executors(int job_index, int count);

  // advance the simulation to the next event (currently runs until idle).
  void run_until_idle();

 private:
  Config cfg_;
  EventLoop events_;
  ExecutorPool pool_;
  std::vector<Job> jobs_;
};

}  // namespace orbit
