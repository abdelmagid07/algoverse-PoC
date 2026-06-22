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

The binding constraint is T4 attention memory (O(tokens^2)). We keep the single
forward pass under MAX_CONTEXT_TOKENS by truncating each observation to
OBS_TOKEN_CAP tokens (reasoning/action `ai` turns are kept intact), and we store
only the step-boundary rows so disk/CPU memory is O(n_steps), not O(seq_len).

Run as a script for a quick standalone collect:
    python -m agent.swebench_loader
"""
from __future__ import annotations

import json
import re

import torch

from interp.activation_cache import RESULTS_DIR, load_model, save_activations

DATASET = "nebius/swe-agent-trajectories"
TRAJ_PATH = RESULTS_DIR / "trajectories.json"

# --- Filtering / budget knobs (set from the Phase A measurements) ----------
# Real trajectories: median 16 ai-steps (p90 32). We keep the 8-20 band so every
# trajectory is long enough to test the early->late forecasting hypothesis while
# staying inside the token budget after observation truncation.
MIN_STEPS = 8
MAX_STEPS = 20

# Observations are the memory hog (p90 ~1158 tok, max ~4000). Capping each to 256
# tokens keeps a median trajectory well under budget; ai turns are left whole.
OBS_TOKEN_CAP = 256
# Conservative T4 budget (15.6 GB). A trajectory still over this after truncation
# is dropped with a warning rather than silently clipped mid-sequence.
MAX_CONTEXT_TOKENS = 8192

N_TRAJECTORIES = 18        # target collection size (roughly balanced)
MAX_SCAN_ROWS = 4000       # cap on streamed rows while hunting for balance

_OBS_ROLES = {"user", "system"}
_STEP_ROLE = "ai"


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
        if role == _STEP_ROLE:
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
    max_per_instance: int = 3,
) -> list[dict]:
    """Stream the dataset and collect ~n trajectories with roughly balanced labels.

    Positives (`target=True`) are rare (~17% in the spike), so we bucket by label
    and try to fill n//2 of each within max_scan_rows. We report the actual
    balance and never fabricate or force it.

    `max_per_instance` caps how many attempts of the SAME SWE-bench task we keep:
    consecutive rows are repeated attempts at one instance, so without this the
    probe could learn instance identity instead of success/failure. We spread the
    sample across distinct tasks instead.
    """
    from datasets import load_dataset

    print(f"Streaming {DATASET} (scan<= {max_scan_rows} rows for "
          f"{n} trajectories, {min_steps}-{max_steps} steps, "
          f"<= {max_per_instance}/instance)...", flush=True)
    ds = load_dataset(DATASET, split="train", streaming=True)

    per_class = max(1, n // 2)
    pos: list[dict] = []
    neg: list[dict] = []
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
        bucket = pos if rec["success"] else neg
        if len(bucket) < per_class:
            bucket.append(rec)
            per_instance[rec["instance_id"]] = per_instance.get(rec["instance_id"], 0) + 1
        if len(pos) >= per_class and len(neg) >= per_class:
            break

    out = pos + neg
    # If one class is short, top up from the other so we still reach ~n usable
    # examples (probe handles imbalance via AUC + majority-class chance baseline).
    if len(out) < n:
        print(f"  note: only {len(pos)} pos / {len(neg)} neg within scan budget; "
              f"reporting actual balance (not forcing).", flush=True)

    n_instances = len({r["instance_id"] for r in out})
    print(f"Collected {len(out)} trajectories "
          f"(success={len(pos)}, fail={len(neg)}) across {n_instances} distinct "
          f"instances after scanning {scanned} rows.", flush=True)
    return out


# --------------------------------------------------------------------------
# Replay + compact activation caching (needs the model).
# --------------------------------------------------------------------------
def _to_ids(model, text: str) -> list[int]:
    if not text:
        return []
    return model.to_tokens(text, prepend_bos=False)[0].tolist()


def build_replay_tokens(model, turns: list[dict]):
    """Concatenate turns into one token sequence, truncating observations.

    Returns (token_ids, step_positions) where step_positions are the absolute
    indices of each `ai` turn's final token. Returns (None, None) if the
    sequence exceeds MAX_CONTEXT_TOKENS even after observation truncation.
    """
    bos = model.tokenizer.bos_token_id
    ids: list[int] = [bos] if bos is not None else []
    step_positions: list[int] = []

    for turn in turns:
        role = turn["role"]
        # A short role tag keeps the replayed text readable as a conversation.
        ids += _to_ids(model, f"\n\n{role}:\n")
        body = _to_ids(model, turn["text"])
        if role in _OBS_ROLES and len(body) > OBS_TOKEN_CAP:
            body = body[:OBS_TOKEN_CAP]  # head-truncate observation (caveat #2)
        ids += body
        if role == _STEP_ROLE:
            step_positions.append(len(ids) - 1)

    if len(ids) > MAX_CONTEXT_TOKENS:
        return None, None
    return ids, step_positions


def replay_and_cache_activations(model, trajectory: dict) -> dict | None:
    """Replay one trajectory and cache step-boundary resid_post for all layers.

    Mutates and returns the trajectory record with the probe-facing fields
    (step_positions=range(n_steps), activation_path, seq_len), and drops `turns`.
    Returns None if the trajectory overflows the token budget (dropped).
    """
    ids, abs_positions = build_replay_tokens(model, trajectory["turns"])
    if ids is None:
        print(f"  warning: {trajectory['id'][:24]} over {MAX_CONTEXT_TOKENS} tokens "
              f"after truncation — dropping.", flush=True)
        return None

    tokens = torch.tensor([ids], device=model.cfg.device)
    names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=lambda nm: nm in names)

    # Keep only step-boundary rows: O(seq) -> O(n_steps) on disk and in RAM.
    idx = torch.tensor(abs_positions, device=model.cfg.device)
    compact = {nm: cache[nm][0].index_select(0, idx).detach().cpu().contiguous()
               for nm in names}
    path = save_activations(trajectory["id"], compact)

    n_steps = len(abs_positions)
    trajectory["step_positions"] = list(range(n_steps))   # probe indexes 0..n-1
    trajectory["n_steps"] = n_steps
    trajectory["seq_len"] = len(ids)
    trajectory["activation_path"] = str(path)
    # Keep short step previews for the qualitative read; drop bulky raw turns.
    trajectory["step_texts"] = [
        t["text"][:200] for t in trajectory["turns"] if t["role"] == _STEP_ROLE
    ][:n_steps]
    trajectory.pop("turns", None)
    return trajectory


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
