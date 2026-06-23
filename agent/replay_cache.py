"""Shared replay tokenization + compact step-boundary activation caching.

Used by both foreign-trajectory replay (swebench_loader) and the live sandbox
agent (sandbox_runner) so the probe sees identical activation semantics.
"""
from __future__ import annotations

import torch

from interp.activation_cache import save_activations

OBS_ROLES = {"user", "system"}
STEP_ROLE = "ai"
OBS_TOKEN_CAP = 256
MAX_CONTEXT_TOKENS = 8192


def max_context_tokens(model) -> int:
    """The model is loaded with a raised n_ctx (8192); honor it as the ceiling."""
    return int(getattr(model.cfg, "n_ctx", MAX_CONTEXT_TOKENS))


def to_ids(model, text: str) -> list[int]:
    if not text:
        return []
    return model.to_tokens(text, prepend_bos=False)[0].tolist()


def build_replay_tokens(model, turns: list[dict]):
    """Concatenate turns into one token sequence, truncating only observations.

    `ai` turns are kept whole; `user`/`system` observations are head-truncated
    to OBS_TOKEN_CAP. Returns (token_ids, step_positions) where step_positions
    are absolute indices of each `ai` turn's final token, or (None, None) if
    the sequence exceeds n_ctx after observation truncation.
    """
    max_tokens = max_context_tokens(model)
    bos = model.tokenizer.bos_token_id
    ids: list[int] = [bos] if bos is not None else []
    step_positions: list[int] = []

    for turn in turns:
        role = turn["role"]
        ids += to_ids(model, f"\n\n{role}:\n")
        body = to_ids(model, turn["text"])
        if role in OBS_ROLES and len(body) > OBS_TOKEN_CAP:
            body = body[:OBS_TOKEN_CAP]
        ids += body
        if role == STEP_ROLE:
            step_positions.append(len(ids) - 1)

    if len(ids) > max_tokens:
        return None, None
    return ids, step_positions


def cache_step_boundary_activations(
    model,
    turns: list[dict],
    traj_id: str,
) -> tuple[dict | None, int, int | None]:
    """Forward pass on turn list; save compact [n_steps, d_model] activations.

    Returns (compact_acts_or_None, n_steps, seq_len). None if over context budget.
    """
    ids, abs_positions = build_replay_tokens(model, turns)
    if ids is None:
        return None, 0, None

    seq_len = len(ids)
    tokens = torch.tensor([ids], device=model.cfg.device)
    names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda nm: nm in names)

    idx = torch.tensor(abs_positions, device=model.cfg.device)
    compact = {
        nm: cache[nm][0].index_select(0, idx).detach().cpu().contiguous()
        for nm in names
    }
    del cache, tokens, idx, ids
    from interp.activation_cache import free_runtime_memory

    save_activations(traj_id, compact)
    free_runtime_memory()
    return compact, len(abs_positions), seq_len


def finalize_trajectory_from_turns(
    model,
    trajectory: dict,
    turns: list[dict],
) -> dict | None:
    """Attach probe-facing fields from a turn list; drop bulky `turns` key."""
    compact, n_steps, seq_len = cache_step_boundary_activations(
        model, turns, trajectory["id"]
    )
    if compact is None:
        print(
            f"  warning: {trajectory['id'][:24]} over {max_context_tokens(model)} "
            f"tokens after observation truncation — dropping.",
            flush=True,
        )
        return None

    from interp.activation_cache import ACT_DIR

    trajectory["step_positions"] = list(range(n_steps))
    trajectory["n_steps"] = n_steps
    trajectory["seq_len"] = seq_len
    trajectory["activation_path"] = str(ACT_DIR / f"{trajectory['id']}.pt")
    trajectory["step_texts"] = [
        t["text"] for t in turns if t["role"] == STEP_ROLE
    ][:n_steps]
    trajectory.pop("turns", None)
    return trajectory
