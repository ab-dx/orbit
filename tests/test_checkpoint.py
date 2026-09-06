from __future__ import annotations

import torch

from orbit.agent import Policy, load_policy, save_policy

N_FEATURES = 12


def test_round_trip_preserves_weights(tmp_path) -> None:
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    path = tmp_path / "pol.pt"
    save_policy(pol, path)
    loaded = load_policy(path)
    for a, b in zip(pol.parameters(), loaded.parameters()):
        assert torch.equal(a, b)


def test_round_trip_preserves_architecture(tmp_path) -> None:
    pol = Policy(in_dim=N_FEATURES, hidden_dim=32, num_layers=3)
    path = tmp_path / "pol.pt"
    save_policy(pol, path)
    loaded = load_policy(path)
    assert loaded.encoder.in_dim == N_FEATURES
    assert loaded.encoder.hidden_dim == 32
    assert loaded.encoder.num_layers == 3
    assert loaded.n_tiles == pol.n_tiles


def test_load_returns_eval_mode(tmp_path) -> None:
    pol = Policy(in_dim=N_FEATURES, hidden_dim=16)
    path = tmp_path / "pol.pt"
    save_policy(pol, path)
    loaded = load_policy(path)
    assert not loaded.training
