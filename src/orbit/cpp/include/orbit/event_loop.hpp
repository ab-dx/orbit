#pragma once

#include <functional>
#include <queue>
#include <vector>

namespace orbit {

// a discrete event loop. callbacks are scheduled at absolute timestamps and
// run in earliest-time order using a min heap.
class EventLoop {
 public:
  using Clock = double;                       // simulated seconds
  using Callback = std::function<void()>;

  Clock now() const { return now_; }

  void schedule_at(Clock t, Callback cb) { queue_.push({t, std::move(cb)}); }

  void schedule_delta(Clock dt, Callback cb) {
    schedule_at(now_ + dt, std::move(cb));
  }

  // run all events with time <= until; returns false if the queue emptied
  // first.
  bool run_until(Clock until);

 private:
  struct Event {
    Clock time;
    Callback cb;
  };
  struct Later {
    bool operator()(const Event& a, const Event& b) const {
      // priority_queue is a max heap, flip the order to pop earliest first.
      return a.time > b.time;
    }
  };

  Clock now_ = 0.0;
  std::priority_queue<Event, std::vector<Event>, Later> queue_;
};

}  // namespace orbit
