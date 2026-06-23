"""Phase 1 (live sandbox): Llama generates multi-turn coding repair trajectories.

Unlike foreign-trajectory replay (agent/swebench_loader.py), Llama-3.2-1B acts
in a lightweight Python sandbox: read/write solution.py, run hidden tests, finish.
Success is live eval (tests pass), not a dataset label.

v2 design: K trajectories per task (different seeds / sampling noise) so labels
are not isomorphic to task identity.

Outputs the same probe contract:
  * results/trajectories.json
  * data/activations/{id}.pt  — compact [n_steps, d_model] at ai step boundaries

Run:  python -m agent.sandbox_runner
"""
from __future__ import annotations

import json
import os
import random
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
from interp.activation_cache import RESULTS_DIR, free_runtime_memory, load_model

TRAJ_PATH = RESULTS_DIR / "trajectories.json"

MIN_STEPS = int(os.environ.get("VERITAS_MIN_STEPS", "8"))
MAX_STEPS = 15
MAX_AGENT_STEPS = 15
MAX_NEW_TOKENS = 320
FREQ_PENALTY = 1.0

N_TASKS_TARGET = int(os.environ.get("VERITAS_N_TASKS", "50"))
K_TRAJECTORIES_PER_TASK = int(os.environ.get("VERITAS_K_PER_TASK", "4"))
TEMPERATURE = float(os.environ.get("VERITAS_TEMPERATURE", "0.7"))

# Legacy alias for pipeline code that still references total trajectory budget.
N_TRAJECTORIES = N_TASKS_TARGET * K_TRAJECTORIES_PER_TASK


def _traj_id(task_id: str, seed: int) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{task_id}__seed{seed}")


def _turns_to_messages(turns: list[dict]) -> list[dict]:
    """Map internal turns (ai) to chat-template roles (assistant)."""
    role_map = {"ai": "assistant", "user": "user", "system": "system"}
    return [{"role": role_map[t["role"]], "content": t["text"]} for t in turns]


def _generate_ai_turn(model, turns: list[dict], *, temperature: float) -> str:
    messages = _turns_to_messages(turns)
    prompt_str = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = model.to_tokens(prompt_str, prepend_bos=False)
    out = model.generate(
        tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=temperature,
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


def run_live_trajectory(
    model,
    task,
    seed: int = 0,
    *,
    temperature: float = TEMPERATURE,
) -> dict | None:
    """Run one multi-turn sandbox episode; return record with `turns` or None."""
    torch.manual_seed(seed)
    random.seed(seed)

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
            ai_text = _generate_ai_turn(model, turns, temperature=temperature)
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
        "seed": seed,
        "temperature": temperature,
        "turns": turns,
        "n_steps": n_ai,
    }
    return record


def load_sandbox_task_list(
    n_tasks: int = N_TASKS_TARGET,
    smoke_n: int | None = None,
) -> list:
    """Shuffled tasks for collection (up to n_tasks)."""
    tasks = load_tasks()
    if smoke_n is not None:
        return tasks[:smoke_n]
    rng = random.Random(0)
    shuffled = tasks.copy()
    rng.shuffle(shuffled)
    return shuffled[: min(n_tasks, len(shuffled))]


def collect_trajectories(
    model,
    n_tasks: int = N_TASKS_TARGET,
    k_per_task: int = K_TRAJECTORIES_PER_TASK,
    smoke_n: int | None = None,
    existing: list[dict] | None = None,
) -> list[dict]:
    """Collect K trajectories per task across n_tasks (v2 within-task design)."""
    if smoke_n is not None:
        n_tasks = smoke_n
        k_per_task = min(3, k_per_task)

    candidate_tasks = load_sandbox_task_list(n_tasks=n_tasks, smoke_n=smoke_n)

    trajectories: list[dict] = list(existing or [])
    done_keys: set[tuple[str, int]] = {
        (t["instance_id"], t.get("seed", 0)) for t in trajectories
    }

    total_target = n_tasks * k_per_task
    print(
        f"Collecting {k_per_task} trajectories x {len(candidate_tasks)} tasks "
        f"(target {total_target} total, step band [{MIN_STEPS},{MAX_STEPS}])",
        flush=True,
    )

    for task in candidate_tasks:
        for seed in range(k_per_task):
            if (task.id, seed) in done_keys:
                continue
            print(
                f"  [{len(trajectories)+1}/{total_target}] "
                f"task={task.id} seed={seed} ...",
                flush=True,
            )
            raw = run_live_trajectory(model, task, seed=seed)
            if raw is None:
                continue
            done = finalize_trajectory_from_turns(model, raw, raw["turns"])
            if done is None:
                continue
            done["seed"] = seed
            done["temperature"] = TEMPERATURE
            trajectories.append(done)
            done_keys.add((task.id, seed))
            TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
            print(
                f"  saved id={done['id'][:24]} steps={done['n_steps']} "
                f"success={done['success']} seq_len={done['seq_len']}",
                flush=True,
            )
            free_runtime_memory()

    return trajectories


def main() -> None:
    smoke = os.environ.get("VERITAS_SMOKE_N")
    smoke_n = int(smoke) if smoke else None

    model = load_model()
    print(
        f"Loaded {model.cfg.model_name} on {model.cfg.device} "
        f"({model.cfg.n_layers} layers, n_ctx={max_context_tokens(model)})",
        flush=True,
    )
    print(
        f"Live sandbox collect v2: {N_TASKS_TARGET} tasks x "
        f"{K_TRAJECTORIES_PER_TASK} seeds, temperature={TEMPERATURE}",
        flush=True,
    )

    trajectories = collect_trajectories(model, smoke_n=smoke_n)
    n_success = sum(t["success"] for t in trajectories)
    print(f"\nSaved {len(trajectories)} trajectories to {TRAJ_PATH}", flush=True)
    print(
        f"Success: {n_success}/{len(trajectories)}  |  "
        f"tasks: {len({t['instance_id'] for t in trajectories})}",
        flush=True,
    )


if __name__ == "__main__":
    main()
