#pragma once

#include "orbit/dag.hpp"
#include "orbit/event_loop.hpp"
#include "orbit/executor.hpp"

#include <cstdint>
#include <random>
#include <vector>

namespace orbit {

// the cluster simulator: an eventloop model of how job dags execute on a pool
// of executors. this is the rl training environment.
class Simulator {
 public:
  struct Config {
    int num_executors = 50;
    uint32_t seed = 1;           // rng seed; used by later phases for noise
    double jvm_startup_delay = 0.0;  // seconds to launch an executor on a job
    double first_wave_slowdown = 1.0;  // multiplier on each stage's first wave
  };

  explicit Simulator(Config cfg);

  // reset to an empty cluster.
  void reset();

  // add a job to the cluster; returns its index.
  int add_job(const Job& job);

  // number of idle executors currently available.
  int num_idle() const { return pool_.num_idle(); }

  // jobs currently in the cluster.
  const std::vector<Job>& jobs() const { return jobs_; }

  EventLoop::Clock now() const { return events_.now(); }

  // grant up to count idle executors to job job_index (respecting its
  // parallelism limit) and start work if a stage is runnable.
  void add_executors(int job_index, int count);

  // the job is done when every stage has completed.
  bool job_done(int job_index) const;

  // advance the simulation to the next event (currently runs until idle).
  void run_until_idle();

 private:
  // true when every one of the stage's parents has completed.
  bool dependencies_met(const Job& job, int stage_index) const;

  // after any state change, assign executors to a runnable stage if possible.
  void pump(int job_index);

  // schedule the next task wave for a stage; wave runs for its duration.
  void start_wave(int job_index, int stage_index);

  // called when a wave finishes; advances tasks and fires dependents.
  void on_wave_done(int job_index, int stage_index, int wave_size);

  // called when a batch of granted executors finishes its jvm startup delay.
  void on_startup_done(int job_index, int count);

  Config cfg_;
  EventLoop events_;
  ExecutorPool pool_;
  std::vector<Job> jobs_;
  std::mt19937 rng_;
};

}  // namespace orbit
