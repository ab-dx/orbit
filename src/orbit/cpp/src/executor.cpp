#include "orbit/executor.hpp"

#include <algorithm>

namespace orbit {

int ExecutorPool::acquire(int requested) {
  const int taken = std::min(requested, num_idle_);
  num_idle_ -= taken;
  return taken;
}

void ExecutorPool::release(int count) {
  const int returned = std::min(count, num_executors_ - num_idle_);
  num_idle_ += returned;
}

int ExecutorPool::available_for(const Job& job) const {
  const int idle = num_idle();
  const int headroom = job.parallelism_limit - job.assigned_executors;
  if (headroom <= 0) return 0;
  return std::min(idle, headroom);
}

}  // namespace orbit
