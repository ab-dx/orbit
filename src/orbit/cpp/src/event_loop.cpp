#include "orbit/event_loop.hpp"

namespace orbit {

bool EventLoop::run_until(Clock until) {
  while (!queue_.empty()) {
    if (queue_.top().time > until) {
      // not due yet; stop, leaving the rest for the caller
      now_ = until;
      return true;
    }
    Event ev = std::move(queue_.top());
    queue_.pop();
    now_ = ev.time;
    ev.cb();
  }
  // queue emptied; now_ already points at the last processed event.
  return false;
}

bool EventLoop::run_next() {
  if (queue_.empty()) return false;
  Event ev = std::move(queue_.top());
  queue_.pop();
  now_ = ev.time;
  ev.cb();
  return true;
}

}  // namespace orbit
