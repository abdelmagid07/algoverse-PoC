"""Phase 1 (live sandbox): Llama generates multi-turn coding repair trajectories.

Unlike foreign-trajectory replay (agent/swebench_loader.py), Llama-3.2-1B acts
in a lightweight Python sandbox: read/write solution.py, run hidden tests, finish.
Success is live eval (tests pass), not a dataset label.

Outputs the same probe contract:
  * results/trajectories.json
  * data/activations/{id}.pt  — compact [n_steps, d_model] at ai step boundaries

Run:  python -m agent.sandbox_runner
"""
from __future__ import annotations

import json
import os
import re

import torch

from agent.replay_cache import build_replay_tokens, finalize_trajectory_from_turns, max_context_tokens
from agent.sandbox_env import (
    SYSTEM_PROMPT,
    Sandbox,
    execute_action,
    initial_user_message,
    load_tasks,
)
from interp.activation_cache import RESULTS_DIR, load_model

TRAJ_PATH = RESULTS_DIR / "trajectories.json"

MIN_STEPS = int(os.environ.get("VERITAS_MIN_STEPS", "8"))
MAX_STEPS = 15
MAX_AGENT_STEPS = 15
MAX_NEW_TOKENS = 320
FREQ_PENALTY = 1.0
N_TRAJECTORIES = 20
MAX_PER_INSTANCE = 1


def _traj_id(task_id: str, seed: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{task_id}__seed{seed}")


def _turns_to_messages(turns: list[dict]) -> list[dict]:
    """Map internal turns (ai) to chat-template roles (assistant)."""
    role_map = {"ai": "assistant", "user": "user", "system": "system"}
    return [{"role": role_map[t["role"]], "content": t["text"]} for t in turns]


def _generate_ai_turn(model, turns: list[dict]) -> str:
    messages = _turns_to_messages(turns)
    prompt_str = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = model.to_tokens(prompt_str, prepend_bos=False)
    out = model.generate(
        tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        freq_penalty=FREQ_PENALTY,
        use_past_kv_cache=True,
        stop_at_eos=True,
        verbose=False,
        return_type="tokens",
    )
    new_tokens = out[0, tokens.shape[1]:]
    return model.to_string(new_tokens).strip()


def parse_action(text: str) -> dict | None:
    """Extract a JSON action object from model output."""
    start = 0
    while True:
        start = text.find("{", start)
        if start == -1:
            break
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict) and obj.get("action"):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start += 1
    # Keyword fallback for small models
    lower = text.lower()
    if "run_tests" in lower or "run tests" in lower:
        return {"action": "run_tests", "thought": text[:120]}
    if "finish" in lower:
        return {"action": "finish", "thought": text[:120]}
    if "read_file" in lower or "read file" in lower:
        return {"action": "read_file", "path": "solution.py", "thought": text[:120]}
    if "write_file" in lower or "write file" in lower:
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
        if m:
            return {
                "action": "write_file",
                "path": "solution.py",
                "content": m.group(1).strip(),
                "thought": text[:120],
            }
    return None


def run_live_trajectory(model, task, seed: int = 0) -> dict | None:
    """Run one multi-turn sandbox episode; return record with `turns` or None."""
    traj_id = _traj_id(task.id, seed)
    sandbox = Sandbox(task, traj_id)

    turns: list[dict] = [
        {"role": "system", "text": SYSTEM_PROMPT},
        {"role": "user", "text": initial_user_message(task)},
    ]
    finished = False
    parse_failures = 0

    try:
        for _ in range(MAX_AGENT_STEPS):
            ai_text = _generate_ai_turn(model, turns)
            if not ai_text:
                break
            turns.append({"role": "ai", "text": ai_text})

            action = parse_action(ai_text)
            if action is None:
                parse_failures += 1
                obs = (
                    "Error: could not parse action. Respond with ONE JSON object, e.g. "
                    '{"thought": "...", "action": "read_file", "path": "solution.py"}'
                )
                if parse_failures >= 2:
                    obs += "\n(Session ending after repeated parse errors.)"
                    turns.append({"role": "user", "text": obs})
                    break
                turns.append({"role": "user", "text": obs})
                continue

            obs = execute_action(sandbox, action)
            turns.append({"role": "user", "text": obs})

            if (action.get("action") or "").lower() == "finish":
                finished = True
                break

            # Early context check — drop before wasting more GPU time.
            ids, _ = build_replay_tokens(model, turns)
            if ids is None:
                print(f"  warning: {traj_id[:24]} exceeded context mid-run — dropping.",
                      flush=True)
                return None
    finally:
        success = sandbox.evaluate_success()
        sandbox.cleanup()

    n_ai = sum(1 for t in turns if t["role"] == "ai")
    if n_ai < MIN_STEPS or n_ai > MAX_STEPS:
        print(f"  skip {traj_id[:24]}: n_steps={n_ai} outside [{MIN_STEPS},{MAX_STEPS}]",
              flush=True)
        return None

    record = {
        "id": traj_id,
        "instance_id": task.id,
        "model_name": model.cfg.model_name,
        "source": "live_sandbox",
        "success": success,
        "finished": finished,
        "turns": turns,
        "n_steps": n_ai,
    }
    return record


def load_sandbox_task_list(
    n: int = N_TRAJECTORIES,
    max_per_instance: int = MAX_PER_INSTANCE,
    smoke_n: int | None = None,
) -> list:
    """Shuffled tasks with at most one attempt per task id."""
    tasks = load_tasks()
    if smoke_n is not None:
        return tasks[:smoke_n]
    # Shuffle deterministically for reproducibility across runs.
    import random
    rng = random.Random(0)
    shuffled = tasks.copy()
    rng.shuffle(shuffled)
    seen: set[str] = set()
    out = []
    for t in shuffled:
        if t.id in seen:
            continue
        if len(seen) >= max_per_instance * n:
            break
        seen.add(t.id)
        out.append(t)
        if len(out) >= n:
            break
    return out


def collect_trajectories(
    model,
    n: int = N_TRAJECTORIES,
    smoke_n: int | None = None,
    existing: list[dict] | None = None,
) -> list[dict]:
    """Collect up to n valid trajectories; may scan extra tasks if steps filter drops some."""
    tasks = load_tasks()
    if smoke_n is not None:
        candidate_tasks = tasks[:smoke_n * 3]  # extra headroom when step filter drops runs
    else:
        import random
        rng = random.Random(0)
        candidate_tasks = tasks.copy()
        rng.shuffle(candidate_tasks)

    trajectories: list[dict] = list(existing or [])
    used_tasks: set[str] = {t["instance_id"] for t in trajectories}
    task_idx = 0

    while len(trajectories) < n and task_idx < len(candidate_tasks):
        task = candidate_tasks[task_idx]
        task_idx += 1
        if task.id in used_tasks:
            continue
        used_tasks.add(task.id)

        print(f"  [{len(trajectories)+1}/{n}] task={task.id} ...", flush=True)
        raw = run_live_trajectory(model, task)
        if raw is None:
            continue
        done = finalize_trajectory_from_turns(model, raw, raw["turns"])
        if done is None:
            continue
        trajectories.append(done)
        TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
        print(
            f"  saved id={done['id'][:24]} steps={done['n_steps']} "
            f"success={done['success']} seq_len={done['seq_len']}",
            flush=True,
        )

    if len(trajectories) < n and smoke_n is None:
        for task in candidate_tasks:
            if len(trajectories) >= n:
                break
            if task.id in used_tasks:
                continue
            used_tasks.add(task.id)
            print(f"  [extra] task={task.id} ...", flush=True)
            raw = run_live_trajectory(model, task)
            if raw is None:
                continue
            done = finalize_trajectory_from_turns(model, raw, raw["turns"])
            if done is None:
                continue
            trajectories.append(done)
            TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    return trajectories


def main() -> None:
    smoke = os.environ.get("VERITAS_SMOKE_N")
    smoke_n = int(smoke) if smoke else None
    n = smoke_n or N_TRAJECTORIES

    model = load_model()
    print(
        f"Loaded {model.cfg.model_name} on {model.cfg.device} "
        f"({model.cfg.n_layers} layers, n_ctx={max_context_tokens(model)})",
        flush=True,
    )
    print(
        f"Live sandbox collect: target={n} trajectories, "
        f"step band [{MIN_STEPS},{MAX_STEPS}]",
        flush=True,
    )

    trajectories = collect_trajectories(model, n=n, smoke_n=smoke_n)
    n_success = sum(t["success"] for t in trajectories)
    print(f"\nSaved {len(trajectories)} trajectories to {TRAJ_PATH}", flush=True)
    print(
        f"Success: {n_success}/{len(trajectories)}  |  "
        f"steps/traj: {[t['n_steps'] for t in trajectories]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
