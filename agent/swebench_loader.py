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
forward pass under `model.cfg.n_ctx` (2048 in TransformerLens) by truncating each
observation and capping each `ai` turn, dropping oldest (user, ai) pairs if still
over budget, and we store
only the step-boundary rows so disk/CPU memory is O(n_steps), not O(seq_len).
Replay must also fit `model.cfg.n_ctx` (2048 in TransformerLens); longer
sequences crash in rotary position encoding even if T4 VRAM would allow them.

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

# Observations are the memory hog (p90 ~1158 tok, max ~4000). Capping each keeps
# replay under the model context window; ai turns are capped too when needed.
OBS_TOKEN_CAP = 64
AI_TOKEN_CAP = 128
# Hard ceiling is model.cfg.n_ctx (2048 for Llama-3.2-1B in TransformerLens).
# We previously used 8192 for T4 VRAM, but rotary embeddings are precomputed only
# up to n_ctx — longer sequences crash in apply_rotary (4819 vs 2048).
MAX_CONTEXT_TOKENS = 2048   # fallback when model is not passed to the encoder

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
def _max_context_tokens(model) -> int:
    """TransformerLens precomputes rotary embeddings only up to n_ctx."""
    return int(getattr(model.cfg, "n_ctx", MAX_CONTEXT_TOKENS))


def _to_ids(model, text: str) -> list[int]:
    if not text:
        return []
    return model.to_tokens(text, prepend_bos=False)[0].tolist()


def _preamble_end(turns: list[dict]) -> int:
    """Index where droppable (user, ai) pairs start — after system + issue user."""
    i = 0
    if turns and turns[0]["role"] == "system":
        i = 1
    if i < len(turns) and turns[i]["role"] == "user":
        i += 1
    return i


def _encode_turns(
    model,
    turns: list[dict],
    obs_cap: int,
    ai_cap: int,
    max_tokens: int,
) -> tuple[list[int], list[int]] | tuple[None, None]:
    """Tokenize turns with per-role caps. Returns (ids, step_positions) or (None, None)."""
    bos = model.tokenizer.bos_token_id
    ids: list[int] = [bos] if bos is not None else []
    step_positions: list[int] = []

    for turn in turns:
        role = turn["role"]
        ids += _to_ids(model, f"\n\n{role}:\n")
        body = _to_ids(model, turn["text"])
        if role in _OBS_ROLES and len(body) > obs_cap:
            body = body[:obs_cap]
        elif role == _STEP_ROLE and len(body) > ai_cap:
            body = body[:ai_cap]
        ids += body
        if role == _STEP_ROLE:
            step_positions.append(len(ids) - 1)

    if len(ids) > max_tokens:
        return None, None
    return ids, step_positions


def build_replay_tokens(model, turns: list[dict]):
    """Concatenate turns into one token sequence, fitting model.cfg.n_ctx.

    Applies observation + ai turn caps, then if still over budget drops the
    oldest (user, ai) pairs after the issue preamble (system + first user).
    Returns (token_ids, step_positions, trimmed_turns, note) or (None, None,
    None, reason) if the trajectory cannot be encoded.
    """
    max_tokens = _max_context_tokens(model)
    working = [{"role": t["role"], "text": t["text"]} for t in turns]
    obs_cap, ai_cap = OBS_TOKEN_CAP, AI_TOKEN_CAP
    dropped_pairs = 0
    preamble = _preamble_end(working)

    while True:
        encoded = _encode_turns(model, working, obs_cap, ai_cap, max_tokens)
        if encoded[0] is not None:
            note = ""
            if dropped_pairs:
                note = f"dropped {dropped_pairs} early user/ai pair(s) to fit n_ctx={max_tokens}"
            if obs_cap < OBS_TOKEN_CAP or ai_cap < AI_TOKEN_CAP:
                note = (note + "; " if note else "") + f"caps obs={obs_cap} ai={ai_cap}"
            return encoded[0], encoded[1], working, note

        # Tighten per-turn caps before dropping history.
        if obs_cap > 32:
            obs_cap = max(32, obs_cap // 2)
            continue
        if ai_cap > 64:
            ai_cap = max(64, ai_cap // 2)
            continue

        # Drop the oldest user+ai pair after the issue preamble.
        if preamble + 2 > len(working):
            return None, None, None, f"cannot fit in n_ctx={max_tokens} even after truncation"

        del working[preamble : preamble + 2]
        dropped_pairs += 1


def replay_and_cache_activations(model, trajectory: dict) -> dict | None:
    """Replay one trajectory and cache step-boundary resid_post for all layers.

    Mutates and returns the trajectory record with the probe-facing fields
    (step_positions=range(n_steps), activation_path, seq_len), and drops `turns`.
    Returns None if the trajectory cannot fit model.cfg.n_ctx (dropped).
    """
    ids, abs_positions, trimmed_turns, note = build_replay_tokens(
        model, trajectory["turns"]
    )
    if ids is None:
        print(f"  warning: {trajectory['id'][:24]} — {note}; dropping.", flush=True)
        return None

    if note:
        print(f"  note: {trajectory['id'][:24]} — {note}", flush=True)

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
    if n_steps < MIN_STEPS:
        print(f"  warning: {trajectory['id'][:24]} only {n_steps} steps after "
              f"truncation (need >={MIN_STEPS}) — dropping.", flush=True)
        return None

    trajectory["step_positions"] = list(range(n_steps))   # probe indexes 0..n-1
    trajectory["n_steps"] = n_steps
    trajectory["seq_len"] = len(ids)
    trajectory["activation_path"] = str(path)
    if note:
        trajectory["truncation_note"] = note
    # Keep short step previews for the qualitative read; drop bulky raw turns.
    trajectory["step_texts"] = [
        t["text"][:200] for t in trimmed_turns if t["role"] == _STEP_ROLE
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
