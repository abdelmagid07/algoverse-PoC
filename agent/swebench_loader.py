"""Phase 1 (SWE-bench): load pre-generated agent trajectories and replay them.

Replaces the live HotpotQA agent loop (agent/runner.py, kept as legacy) as the
trajectory source. We do NOT run an agent here: we read existing SWE-bench agent
runs from `nebius/swe-agent-trajectories`, replay each one's text through
Llama-3.2-1B, and cache the residual stream at step boundaries. The probe code
downstream is unchanged — this module only has to reproduce the probe's data
contract:

  * results/trajectories.json : list of {id, step_positions, success, ...}
  * data/activations/{id}.pt  : {f"blocks.{l}.hook_resid_post": [n_steps, d_model]}
    where step_positions == range(n_steps).

Schema (confirmed in the Phase A spike on real rows):
  row = {
    instance_id: str,            # SWE-bench task (repeats across attempts)
    model_name: str,             # agent that generated the run
    target: bool,                # SUCCESS LABEL (resolved / correct attempt)
    trajectory: [ {role, text, ...} ],   # role in {system, user, ai}
    exit_status, generated_patch, eval_logs: str,   # metadata
  }

Conventions (flagged choices, reported in the summary):
  * STEP BOUNDARY: one `ai` (assistant) turn = one step; its step position is the
    last token of that turn. `user` turns are observations (tool output, issue
    text); `system` is the system prompt. Observations are context, not steps.
  * SUCCESS LABEL: `target` (bool).
  * The agent that produced these runs is NOT Llama-3.2-1B — we replay foreign
    trajectories through our model and read its internal state while it reads the
    run. (Caveat #3 in the summary.)

The model is loaded with n_ctx raised to 8192 (interp/activation_cache.py;
TransformerLens otherwise caps Llama-3.2 at 2048 and longer sequences crash in
rotary position encoding). That fits whole 8-20 step trajectories, so we only
head-truncate `user`/`system` observations to OBS_TOKEN_CAP — `ai` reasoning
turns (what the probe reads) are kept intact and no steps are dropped. We store
only the step-boundary rows so disk/CPU memory is O(n_steps), not O(seq_len).

Run as a script for a quick standalone collect:
    python -m agent.swebench_loader
"""
from __future__ import annotations

import json
import re

import torch

from agent.replay_cache import (
    OBS_TOKEN_CAP,
    STEP_ROLE,
    build_replay_tokens,
    finalize_trajectory_from_turns,
    max_context_tokens,
)
from interp.activation_cache import RESULTS_DIR, load_model

DATASET = "nebius/swe-agent-trajectories"
TRAJ_PATH = RESULTS_DIR / "trajectories.json"

# --- Filtering / budget knobs (set from the Phase A measurements) ----------
# Real trajectories: median 16 ai-steps (p90 32). We keep the 8-20 band so every
# trajectory is long enough to test the early->late forecasting hypothesis while
# staying inside the token budget after observation truncation.
MIN_STEPS = 8
MAX_STEPS = 20

N_TRAJECTORIES = 18        # target collection size
MAX_SCAN_ROWS = 4000       # cap on streamed rows while hunting for balance
MAX_PER_INSTANCE = int(__import__("os").environ.get("VERITAS_K_PER_TASK", "4"))

_OBS_ROLES = {"user", "system"}


# --------------------------------------------------------------------------
# Parsing (no model needed) — cheap, used for filtering + balancing.
# --------------------------------------------------------------------------
def _safe_id(instance_id: str, model_name: str, idx: int) -> str:
    raw = f"{instance_id}__{model_name}__{idx}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def parse_row(row: dict, idx: int,
              min_steps: int = MIN_STEPS, max_steps: int = MAX_STEPS) -> dict | None:
    """Parse one dataset row into an unreplayed trajectory record, or None.

    Returns {id, instance_id, model_name, success, turns, n_steps} where `turns`
    is the ordered [{role, text}] list. Filtered to the [min_steps, max_steps]
    step band. None if it has no usable steps or falls outside the band.
    """
    traj = row.get("trajectory") or []
    turns: list[dict] = []
    n_steps = 0
    for t in traj:
        role = t.get("role")
        text = (t.get("text") or "").strip()
        if role == STEP_ROLE:
            if not text:  # skip empty assistant turns so step positions stay real
                continue
            n_steps += 1
        elif role not in _OBS_ROLES:
            continue
        turns.append({"role": role, "text": text})

    if n_steps < min_steps or n_steps > max_steps:
        return None

    instance_id = str(row.get("instance_id", f"unknown_{idx}"))
    model_name = str(row.get("model_name", "unknown"))
    return {
        "id": _safe_id(instance_id, model_name, idx),
        "instance_id": instance_id,
        "model_name": model_name,
        "success": bool(row.get("target")),
        "turns": turns,
        "n_steps": n_steps,
    }


def load_swebench_trajectories(
    n: int = N_TRAJECTORIES,
    min_steps: int = MIN_STEPS,
    max_steps: int = MAX_STEPS,
    max_scan_rows: int = MAX_SCAN_ROWS,
    max_per_instance: int = MAX_PER_INSTANCE,
) -> list[dict]:
    """Stream the dataset and collect ~n trajectories across distinct instances.

    No manual label balancing — mixed outcomes are filtered post-load via
    analysis.dataset_validate.filter_mixed_tasks().
    """
    from datasets import load_dataset

    print(f"Streaming {DATASET} (scan<= {max_scan_rows} rows for "
          f"{n} trajectories, {min_steps}-{max_steps} steps, "
          f"<= {max_per_instance}/instance)...", flush=True)
    ds = load_dataset(DATASET, split="train", streaming=True)

    collected: list[dict] = []
    per_instance: dict[str, int] = {}
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned > max_scan_rows:
            break
        rec = parse_row(row, scanned, min_steps, max_steps)
        if rec is None:
            continue
        if per_instance.get(rec["instance_id"], 0) >= max_per_instance:
            continue
        if len(collected) >= n:
            break
        collected.append(rec)
        per_instance[rec["instance_id"]] = per_instance.get(rec["instance_id"], 0) + 1

    out = collected

    n_success = sum(1 for r in out if r["success"])
    n_instances = len({r["instance_id"] for r in out})
    print(f"Collected {len(out)} trajectories "
          f"(success={n_success}, fail={len(out) - n_success}) across {n_instances} distinct "
          f"instances after scanning {scanned} rows.", flush=True)
    return out


# --------------------------------------------------------------------------
# Replay + compact activation caching (needs the model).
# --------------------------------------------------------------------------
def replay_and_cache_activations(model, trajectory: dict) -> dict | None:
    """Replay one trajectory and cache step-boundary resid_post for all layers."""
    return finalize_trajectory_from_turns(model, trajectory, trajectory["turns"])


def main() -> None:
    model = load_model()
    print(f"Loaded {model.cfg.model_name} on {model.cfg.device} "
          f"({model.cfg.n_layers} layers, d_model={model.cfg.d_model})", flush=True)

    raw = load_swebench_trajectories()
    trajectories: list[dict] = []
    for i, rec in enumerate(raw):
        print(f"  [{i+1}/{len(raw)}] replay id={rec['id'][:24]} "
              f"(n_steps={rec['n_steps']}, success={rec['success']}) ...", flush=True)
        done = replay_and_cache_activations(model, rec)
        if done is None:
            continue
        trajectories.append(done)
        TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    n_success = sum(t["success"] for t in trajectories)
    print(f"\nSaved {len(trajectories)} trajectories to {TRAJ_PATH}", flush=True)
    print(f"Success: {n_success}/{len(trajectories)}  |  "
          f"steps/traj: {[t['n_steps'] for t in trajectories]}", flush=True)


if __name__ == "__main__":
    main()
