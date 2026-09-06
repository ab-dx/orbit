"""save and load a trained policy from disk."""

from __future__ import annotations

from pathlib import Path

import torch

from .policy import Policy


def save_policy(policy: Policy, path: str | Path) -> None:
    """persist the policy weights plus its architecture args."""
    tiles = tuple(policy.tiles[:-1])  # drop the trailing no-op tile
    payload = {
        "in_dim": policy.encoder.in_dim,
        "hidden_dim": policy.encoder.hidden_dim,
        "num_layers": policy.encoder.num_layers,
        "tiles": tiles,
        "state_dict": policy.state_dict(),
    }
    torch.save(payload, str(path))


def load_policy(path: str | Path) -> Policy:
    """rebuild a policy from a checkpoint written by save_policy."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    policy = Policy(
        in_dim=payload["in_dim"],
        hidden_dim=payload["hidden_dim"],
        num_layers=payload["num_layers"],
        tiles=payload["tiles"],
    )
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy


__all__ = ["load_policy", "save_policy"]
