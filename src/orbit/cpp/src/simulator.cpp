#include "orbit/simulator.hpp"

#include <algorithm>

namespace orbit {

Simulator::Simulator(Config cfg)
    : cfg_(cfg), pool_(cfg.num_executors), rng_(cfg.seed) {}

void Simulator::reset() {
  jobs_.clear();
  pool_.reset();
  // fresh event loop (priority_queue has no clear; reassign to rebuild)
  events_ = EventLoop{};
}

int Simulator::add_job(const Job& job) {
  jobs_.push_back(job);
  const int idx = static_cast<int>(jobs_.size()) - 1;
  pump(idx);
  return idx;
}

void Simulator::add_executors(int job_index, int count) {
  if (job_index < 0 || job_index >= static_cast<int>(jobs_.size())) return;
  Job& job = jobs_[static_cast<size_t>(job_index)];
  if (job.completed) return;

  const int headroom = job.parallelism_limit - job.assigned_executors;
  const int to_take = std::min({count, headroom, pool_.num_idle()});
  const int taken = pool_.acquire(to_take);
  if (taken <= 0) return;

  job.assigned_executors += taken;
  job.starting_executors += taken;

  // granted executors only work after the jvm startup delay
  events_.schedule_delta(cfg_.jvm_startup_delay,
                         [this, job_index, taken]() {
                           on_startup_done(job_index, taken);
                         });
  // existing active executors (if any) keep working; no pump needed yet
}

void Simulator::on_startup_done(int job_index, int count) {
  if (job_index < 0 || job_index >= static_cast<int>(jobs_.size())) return;
  Job& job = jobs_[static_cast<size_t>(job_index)];
  if (job.completed) return;

  const int promoted = std::min(count, job.starting_executors);
  job.starting_executors -= promoted;
  job.active_executors += promoted;
  pump(job_index);
}

bool Simulator::job_done(int job_index) const {
  return job_index >= 0 && job_index < static_cast<int>(jobs_.size()) &&
         jobs_[static_cast<size_t>(job_index)].completed;
}

bool Simulator::dependencies_met(const Job& job, int stage_index) const {
  const auto& st = job.stages[static_cast<size_t>(stage_index)];
  for (const int p : st.parents) {
    if (!job.stages[static_cast<size_t>(p)].completed) return false;
  }
  return true;
}

void Simulator::pump(int job_index) {
  Job& job = jobs_[static_cast<size_t>(job_index)];
  if (job.completed || job.active_executors <= 0) return;

  // already working a stage: sync its executors so the next wave uses the new
  // count, and let the running wave finish.
  if (job.active_stage >= 0) {
    job.stages[static_cast<size_t>(job.active_stage)].assigned_executors =
        job.active_executors;
    return;
  }

  // otherwise pick the next runnable stage and start it
  for (int s = 0; s < static_cast<int>(job.stages.size()); ++s) {
    Stage& st = job.stages[static_cast<size_t>(s)];
    if (st.completed || st.tasks_remaining <= 0 || !dependencies_met(job, s)) {
      continue;
    }
    st.assigned_executors = job.active_executors;
    job.active_stage = s;
    start_wave(job_index, s);
    return;
  }
}

void Simulator::start_wave(int job_index, int stage_index) {
  Job& job = jobs_[static_cast<size_t>(job_index)];
  Stage& st = job.stages[static_cast<size_t>(stage_index)];

  const int wave = std::min(st.assigned_executors, st.tasks_remaining);
  if (wave <= 0) {
    // nothing schedulable right now; free the stage role
    st.assigned_executors = 0;
    job.active_stage = -1;
    return;
  }

  // a wave of wave tasks completes at the stage's effective rate; away
  // from saturation this reduces to wave / (k / avg) = avg per full wave.
  const int k = st.assigned_executors;
  double rate = stage_rate(st, k);
  double duration = wave / rate;
  // the first wave of a stage is slower (jit warmup and task setup)
  if (st.wave_count == 0) {
    duration *= cfg_.first_wave_slowdown;
  }
  st.wave_count += 1;

  events_.schedule_delta(
      duration, [this, job_index, stage_index, wave]() {
        on_wave_done(job_index, stage_index, wave);
      });
}

double Simulator::stage_rate(const Stage& st, int k) const {
  const double lambda = cfg_.executor_rate > 0.0
                            ? cfg_.executor_rate
                            : 1.0 / st.avg_task_duration;
  const double linear = lambda * k;
  return st.mu_cap > 0.0 ? std::min(linear, st.mu_cap) : linear;
}

void Simulator::on_wave_done(int job_index, int stage_index, int wave_size) {
  Job& job = jobs_[static_cast<size_t>(job_index)];
  Stage& st = job.stages[static_cast<size_t>(stage_index)];

  st.tasks_remaining -= wave_size;
  if (st.tasks_remaining > 0) {
    // another wave of the same stage; keep executors and schedule the next
    start_wave(job_index, stage_index);
    return;
  }

  // stage complete: free it and unlock dependents
  st.completed = true;
  st.assigned_executors = 0;
  job.active_stage = -1;

  bool all_done = true;
  for (const auto& s : job.stages) {
    if (!s.completed) {
      all_done = false;
      break;
    }
  }
  if (all_done) {
    // the job is finished; return its executors to the idle pool
    job.completed = true;
    job.completion_time = now();
    pool_.release(job.assigned_executors);
    job.assigned_executors = 0;
    job.starting_executors = 0;
    job.active_executors = 0;
    // record the completion for the reward signal of the current step
    step_completed_ += 1;
    step_jct_ += job.completion_time - job.arrival_time;
    return;
  }
  pump(job_index);
}

bool Simulator::step() {
  // each step covers one decision interval: process events up to and
  // including the next scheduled event, accumulating jct for completions.
  step_completed_ = 0;
  step_jct_ = 0.0;
  if (events_.empty()) return false;
  return events_.run_next();
}

void Simulator::run_until_idle() {
  while (step()) {
  }
}

}  // namespace orbit
