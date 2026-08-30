#include "orbit/dag.hpp"

namespace orbit {

// stage and job are header-only data structures for now. this translation unit
// exists so later phases can add dag operations (topo sort, critical path,
// runnable-stage tracking) here.

}  // namespace orbit
