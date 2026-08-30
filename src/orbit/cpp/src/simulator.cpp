#include "orbit/simulator.hpp"

#include <algorithm>

namespace orbit {

Simulator::Simulator(Config cfg) : cfg_(cfg), pool_(cfg.num_executors) {}

void Simulator::reset() {
  jobs_.clear();
  // todo: clear pool assignment state and the event queue with the full
  // scheduling model.
}

void Simulator::add_executors(int job_index, int count) {
  if (job_index < 0 || job_index >= static_cast<int>(jobs_.size())) return;
  Job& job = jobs_[static_cast<size_t>(job_index)];
  const int headroom = job.parallelism_limit - job.assigned_executors;
  const int granted = std::min(count, std::max(headroom, 0));
  job.assigned_executors += granted;
  // todo: adopt the granted executors from the idle pool and schedule task
  // wave events on the EventLoop.
}

void Simulator::run_until_idle() {
  events_.run_until(events_.now() + 1e9);
}

}  // namespace orbit
