#include "orbit/executor.hpp"

#include <algorithm>

namespace orbit {

int ExecutorPool::available_for(const Job& job) const {
  const int idle = num_idle();
  const int headroom = job.parallelism_limit - job.assigned_executors;
  if (headroom <= 0) return 0;
  return std::min(idle, headroom);
}

}  // namespace orbit
