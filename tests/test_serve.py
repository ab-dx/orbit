from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import grpc
import torch

from orbit import core
from orbit.agent import Policy, graph_input, load_policy, save_policy
from orbit.serve import (
    SchedulerClient,
    SchedulerServicer,
    build_request,
    scheduler_pb2_grpc,
    serve,
)
from orbit.sim import ALLOC_TILES, OrbitEnv, WorkloadConfig

N_FEATURES = 12


def _env() -> OrbitEnv:
    return OrbitEnv(core.SimulatorConfig())


def _wcfg() -> WorkloadConfig:
    return WorkloadConfig(
        job_arrival_rate=1.0,
        num_jobs=3,
        cluster_size=8,
        max_stages=3,
        min_tasks_per_stage=1,
        max_tasks_per_stage=4,
        min_avg_task_duration=0.5,
        max_avg_task_duration=1.5,
    )


def _policy(**kw) -> Policy:
    return Policy(in_dim=N_FEATURES, hidden_dim=16, **kw)


def _start_client(policy: Policy):
    """spin up a server and return (client, server, channel) so the server
    stays alive for the whole test."""
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    scheduler_pb2_grpc.add_SchedulerServicer_to_server(SchedulerServicer(policy), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=10)
    return SchedulerClient(channel), server, channel


def test_served_decision_matches_local_greedy() -> None:
    env = _env()
    policy = _policy()
    client, _server, channel = _start_client(policy)
    obs, info = env.reset(options={"workload": _wcfg(), "seed": 0})
    # step until runnable, then compare the served decision to a local act
    for _ in range(50):
        if info["runnable"]:
            break
        obs, _r, _t, _tr, info = env.step((0, 0))
    assert info["runnable"]
    req = build_request(obs, info)
    resp = client.decide(req)
    with torch.no_grad():
        h = policy.encode(*graph_input(obs))
    action = policy.act(h, info["runnable_node_ids"], info["available"])
    assert resp.stage_action == action.stage
    assert resp.alloc_action == action.alloc
    assert resp.stage_node_id == info["runnable_node_ids"][action.stage]
    channel.close()


def test_served_decisions_stay_legal_over_episode() -> None:
    env = _env()
    policy = _policy()
    client, _server, channel = _start_client(policy)
    obs, info = env.reset(options={"workload": _wcfg(), "seed": 1})
    served = 0
    for _ in range(200):
        if info["terminated"]:
            break
        if info["runnable"]:
            resp = client.decide(build_request(obs, info))
            assert 0 <= resp.stage_action < len(info["runnable"])
            # the alloc index may be the no-op tile, which grants zero
            assert 0 <= resp.alloc_action <= len(ALLOC_TILES)
            if resp.alloc_action < len(ALLOC_TILES):
                assert ALLOC_TILES[resp.alloc_action] <= info["available"][resp.stage_action]
            step_action = (resp.stage_action, resp.alloc_action)
            served += 1
        else:
            step_action = (0, 0)
        obs, _r, _t, _tr, info = env.step(step_action)
    assert served > 0, "no served decision exercised over the episode"
    channel.close()


def test_served_decision_is_deterministic() -> None:
    env = _env()
    policy = _policy()
    client, _server, channel = _start_client(policy)
    obs, info = env.reset(options={"workload": _wcfg(), "seed": 2})
    for _ in range(50):
        if info["runnable"]:
            break
        obs, _r, _t, _tr, info = env.step((0, 0))
    assert info["runnable"]
    req = build_request(obs, info)
    a = client.decide(req)
    b = client.decide(req)
    assert (a.stage_action, a.alloc_action) == (b.stage_action, b.alloc_action)
    channel.close()


def test_serve_helper_binds_and_serves(tmp_path) -> None:
    env = _env()
    policy = _policy()
    path = str(tmp_path / "policy.pt")
    save_policy(policy, path)
    loaded = load_policy(path)
    server, bound = serve(loaded, port=0)
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{bound}")
    grpc.channel_ready_future(channel).result(timeout=10)
    client = SchedulerClient(channel)
    obs, info = env.reset(options={"workload": _wcfg(), "seed": 3})
    for _ in range(50):
        if info["runnable"]:
            break
        obs, _r, _t, _tr, info = env.step((0, 0))
    assert info["runnable"]
    resp = client.decide(build_request(obs, info))
    assert resp.stage_node_id in info["runnable_node_ids"]
    channel.close()
    server.stop(0)
