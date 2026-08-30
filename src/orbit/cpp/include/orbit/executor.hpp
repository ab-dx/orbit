#pragma once

#include "orbit/dag.hpp"

namespace orbit {

// tracks the m homogeneous executor slots. an executor is idle (available) or
// assigned to a job; moving one between jobs costs a jvm startup delay.
class ExecutorPool {
 public:
  explicit ExecutorPool(int num_executors) : num_executors_(num_executors) {}

  int num_executors() const { return num_executors_; }
  int num_idle() const { return num_idle_; }

  // free slots grantable to a job without exceeding its per-job limit. this
  // backs the available executors feature.
  int available_for(const Job& job) const;

 private:
  int num_executors_;
  int num_idle_ = 0;
};

}  // namespace orbit
