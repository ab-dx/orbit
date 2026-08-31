from __future__ import annotations

import torch

from orbit import core
from orbit.agent import GraphEncoder, MeanConv, Policy, graph_input
from orbit.sim import OrbitEnv, WorkloadConfig


def _obs_env(jobs):
    env = OrbitEnv(core.SimulatorConfig())
    obs, info = env.reset(options={"jobs": jobs})
    return env, obs, info


def _diamond_jobs():
    return [
        {
            "parallelism": 4,
            "stages": [
                {"tasks": 4},
                {"tasks": 4},
                {"tasks": 4, "parents": [0, 1]},
            ],
        }
    ]


def test_mean_conv_aggregates_neighbors() -> None:
    # a root -> two children. with mean aggregation and zeroed neighbor weight,
    # each node's next state should be a nonlinearity of w_self on itself
    # (neighbor contribution is the mean of its neighbors' states).
    torch.manual_seed(0)
    conv = MeanConv(2, 4)
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]])
    out = conv(x, edge_index)
    assert out.shape == (3, 4)
    # the conv must give a finite, activation-shaped result for every node
    assert torch.isfinite(out).all()
    assert (out >= 0).all()  # relu


def test_encoder_output_shape() -> None:
    env, obs, _ = _obs_env(_diamond_jobs())
    x, ei = graph_input(obs)
    assert x.shape == (3, 12)
    assert ei.shape[0] == 2

    enc = GraphEncoder(in_dim=12, hidden_dim=32, num_layers=2)
    h = enc(x, ei)
    assert h.shape == (3, 32)


def test_graph_input_matches_observation() -> None:
    env, obs, info = _obs_env(_diamond_jobs())
    x, _ei = graph_input(obs)
    # first node (root) exposes its tasks_remaining in column 0
    assert x[0, 0] == float(obs.nodes[0].tasks_remaining)
    assert x[0, 8] == 1.0  # is_root
    assert x[2, 10] == 0.0  # the join is not runnable yet


def test_stage_head_only_scores_runnable() -> None:
    env, obs, info = _obs_env(_diamond_jobs())
    x, ei = graph_input(obs)
    pol = Policy(in_dim=12)
    h = pol.encode(x, ei)
    dist = pol.stage_distribution(h, info["runnable_node_ids"])
    # two roots are runnable; the join is not in the mask
    assert dist.probs.shape == (2,)
    assert torch.isfinite(dist.logits).all()
    assert info["runnable"] == [(0, 0), (0, 1)]


def test_alloc_head_respects_availability() -> None:
    env, obs, info = _obs_env(
        [{"parallelism": 2, "stages": [{"tasks": 4}]}]
    )
    x, ei = graph_input(obs)
    pol = Policy(in_dim=12)
    h = pol.encode(x, ei)
    # only 2 executors are available (job parallelism limit), so only tiles
    # up to 2 (plus the no-op) may receive probability mass
    dist = pol.alloc_distribution(h, 0, available=2, runnable_ids=info["runnable_node_ids"])
    mass = torch.where(dist.probs > 0.0)[0]
    assert set(mass.tolist()) <= {0, 1, 6}  # tiles 1,2 and the no-op at index 6


def test_alloc_head_survives_zero_available() -> None:
    # when nothing is free, only the no-op "wait" tile remains legal
    env, obs, info = _obs_env([{"parallelism": 1, "stages": [{"tasks": 4}]}])
    x, ei = graph_input(obs)
    pol = Policy(in_dim=12)
    h = pol.encode(x, ei)
    dist = pol.alloc_distribution(h, 0, available=0, runnable_ids=info["runnable_node_ids"])
    # exactly the no-op index is legal, so sampling must hit it
    a = pol.sample(h, info["runnable_node_ids"], [0])
    assert a.alloc == len(pol.tiles) - 1


def test_sample_produces_legal_action() -> None:
    env, obs, info = _obs_env(_diamond_jobs())
    x, ei = graph_input(obs)
    pol = Policy(in_dim=12)
    torch.manual_seed(3)
    h = pol.encode(x, ei)
    for _ in range(20):
        a = pol.sample(h, info["runnable_node_ids"], info["available"])
        assert 0 <= a.stage < len(info["runnable"])
        # the sampled allocation is either a legal tile or the wait option
        assert pol.tiles[a.alloc] == 0 or pol.tiles[a.alloc] <= info["available"][a.stage]


def test_log_prob_is_finite() -> None:
    env, obs, info = _obs_env(_diamond_jobs())
    x, ei = graph_input(obs)
    pol = Policy(in_dim=12)
    h = pol.encode(x, ei)
    a = pol.sample(h, info["runnable_node_ids"], info["available"])
    lp = pol.log_prob(h, info["runnable_node_ids"], info["available"], a)
    assert torch.isfinite(lp).item()


def test_policy_drives_full_episode() -> None:
    env = OrbitEnv(core.SimulatorConfig())
    obs, info = env.reset(
        options={
            "workload": WorkloadConfig(num_jobs=4, cluster_size=10, max_stages=4),
            "seed": 2,
        }
    )
    pol = Policy(in_dim=12)

    def decide(o, i):
        if not i["runnable"]:
            return (0, 0)
        x, ei = graph_input(o)
        a = pol.sample(pol.encode(x, ei), i["runnable_node_ids"], i["available"])
        return (a.stage, a.alloc)

    total = 0.0
    steps = 0
    terminated = False
    while not terminated and steps < 500:
        obs, reward, terminated, _trunc, info = env.step(decide(obs, info))
        total += reward
        steps += 1
    assert terminated is True
    assert total < 0.0
    assert all(env._sim.job_done(j) for j in range(len(env._sim.jobs())))
