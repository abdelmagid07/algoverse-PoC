"""Phase 1: trajectory trace logging.

For each collected trajectory, cache the residual-stream activation at every
layer over the full sequence and persist it to disk, alongside the step
metadata and success/fail label already on the trajectory record. The cached
tensors back the Phase-1 checkpoint ("residual stream activations cached for
every step, every layer") and are available for EDA.
"""
from __future__ import annotations

import torch

from interp.activation_cache import cache_residual_stream, save_activations


def log_trajectory(model, trajectory: dict) -> None:
    """Cache the residual stream for one trajectory and save it to disk."""
    tokens = torch.tensor([trajectory["token_ids"]], device=model.cfg.device)
    acts = cache_residual_stream(model, tokens)
    path = save_activations(trajectory["id"], acts)
    trajectory["activation_path"] = str(path)
    trajectory["seq_len"] = tokens.shape[1]
