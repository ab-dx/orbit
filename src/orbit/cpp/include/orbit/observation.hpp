#pragma once

#include <utility>
#include <vector>

namespace orbit {

// a snapshot of one stage (node) at a decision point, used to build the graph
// encoder's node feature matrix. plain aggregate; no tensor types.
struct StageObs {
  int node_id = -1;          // global node id across all jobs
  int job_index = -1;        // owning job
  int stage_index = -1;      // index within the owning job
  int tasks_remaining = 0;
  double avg_task_duration = 1.0;
  int assigned_executors = 0;
  double mu_cap = 0.0;
  int parallelism_limit = 1;  // owner job's limit
  int headroom = 0;           // parallelism_limit - assigned_executors
  int wave_count = 0;
  int completed = 0;          // 0/1
  int is_root = 0;            // 0/1
  int is_leaf = 0;            // 0/1
  int runnable = 0;           // 0/1
  double age = 0.0;           // seconds since the job arrived
};

// the full observation returned at a decision point: node features for every
// stage across every job, the runnable-stage set (the action mask), and a
// global adjacency list over node ids.
struct Observation {
  std::vector<StageObs> nodes;
  std::vector<std::pair<int, int>> runnable;  // (job_index, stage_index)
  std::vector<int> edge_src;  // source node ids of dag edges
  std::vector<int> edge_dst;  // destination node ids of dag edges (children)
};

}  // namespace orbit
