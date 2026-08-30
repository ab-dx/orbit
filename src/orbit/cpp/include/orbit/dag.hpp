#pragma once

#include <vector>

namespace orbit {

// a stage (a node) in a job's dag. holds the per-node features the graph
// encoder reads: tasks remaining, average task duration, executors on the
// node, available executors, and whether those are local to the job.
struct Stage {
  std::vector<int> parents;  // indices into the owning job's stage array
  std::vector<int> children;

  int total_tasks = 0;
  int tasks_remaining = 0;
  double avg_task_duration = 1.0;  // seconds, from profiling
  int assigned_executors = 0;      // executors currently working on this stage
  bool completed = false;          // all tasks done, dependents may fire

  bool is_root() const { return parents.empty(); }
  bool is_leaf() const { return children.empty(); }
  bool runnable() const { return tasks_remaining > 0; }
};

// a job is a dag of stages plus a parallelism limit, the max number of
// executors that may be assigned to this job at once.
struct Job {
  std::vector<Stage> stages;
  int parallelism_limit = 1;   // l_i
  int assigned_executors = 0;  // current executor count for this job
  double arrival_time = 0.0;

  // runtime scheduling state, owned by the simulator
  int active_stage = -1;      // stage being worked on, -1 if none
  bool completed = false;     // every stage finished
  double completion_time = -1.0;
};

}  // namespace orbit
