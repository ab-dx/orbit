# Orbit

A reinforcement-learning cluster scheduler, built for learning the underlying
systems + RL concepts from the ground up.

The scheduler uses a graph neural network over job dags and a REINFORCE
policy-gradient agent to learn scheduling policies, choosing which stage to run
next and how much parallelism each job gets, with the goal of minimizing
average job completion time (jct).

## architecture

```
src/orbit/
├── cpp/                  c++17 core (performance-critical)
│   ├── include/orbit/     headers (dag.hpp, executor.hpp, event_loop.hpp, simulator.hpp)
│   ├── src/               implementations + pybind11 bindings
│   └── CMakeLists.txt     builds _orbit_core extension
├── __init__.py            exposes orbit.core (compiled module)
├── cli.py                 `orbit` console command
├── sim/                   gymnasium environment wrapping the c++ simulator
├── agent/                 gnn state encoder + policy network
├── train/                 reinforce training loop
└── serve/                 grpc inference server (deployment path)
tests/                     pytest smoke tests
```

the simulator runs as a c++17 discrete-event engine exposed to python via
pybind11 and built with scikit-build-core. the ml/rl stack (pytorch,
gnn, reinforce) lives in python.

## setup

```sh
uv venv --python 3.12 .venv
uv pip install -e . --python .venv/bin/python   # builds the c++ extension
uv pip install -e ".[dev]" --python .venv/bin/python
.venv/bin/python -m pytest                       # run smoke tests
.venv/bin/orbit                                  # verify cli + core
```

> when you edit c++ sources, reinstall to trigger a rebuild:
> `uv pip install -e . --no-deps --python .venv/bin/python`

## roadmap

- phase 0: project scaffolding, c++ build pipeline + smoke test.
- phase 1: discrete-event dag simulator, task waves, jvm startup delay,
  high-parallelism slowdown, scheduling actions, gymnasium env.
- phase 2: synthetic dag + poisson-arrival workload generator.
- phase 3: gnn state encoder + policy network (stage + parallelism heads).
- phase 4: reinforce training (differential rewards, fixed-sequence
  baselines, curriculum episode length).
- phase 5: evaluation vs heuristics (fifo, sjf-cp, fair, weighted fair).
- phase 6: grpc inference server (spark integration as a stretch goal).
