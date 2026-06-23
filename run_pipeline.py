"""Run the full Latent Failure Forecasting PoC pipeline in one process.

Pipeline v2: collect K trajectories per task (live sandbox) -> validate ->
probe (within-task + LOTO) -> skepticism -> visualize -> summarize.

Set VERITAS_TRAJECTORY_SOURCE=replay to use legacy SWE-bench replay.

Usage (Colab or local):
    python run_pipeline.py
    VERITAS_SMOKE_N=2 python run_pipeline.py   # smoke: 2 tasks x 3 seeds
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dataset_validate import validate_dataset  # noqa: E402
from analysis.probe import main as probe_main  # noqa: E402
from analysis.skepticism import main as skepticism_main  # noqa: E402
from analysis.run_scorer import main as score_main  # noqa: E402
from analysis.summarize import main as summarize_main  # noqa: E402
from analysis.visualize_probe import main as visualize_main  # noqa: E402
from interp.activation_cache import RESULTS_DIR, MODEL_NAME, load_model, log_device_choice  # noqa: E402

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
TRAJECTORY_SOURCE = os.environ.get("VERITAS_TRAJECTORY_SOURCE", "live").lower()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _activation_exists(traj: dict) -> bool:
    if "activation_path" not in traj:
        return False
    return Path(traj["activation_path"]).exists()


def phase1_collect_live(model) -> list[dict]:
    from agent.sandbox_runner import (
        K_TRAJECTORIES_PER_TASK,
        N_TASKS_TARGET,
        collect_trajectories,
    )

    smoke = os.environ.get("VERITAS_SMOKE_N")
    smoke_n = int(smoke) if smoke else None
    n_tasks = smoke_n if smoke_n else N_TASKS_TARGET
    k_per = 3 if smoke_n else K_TRAJECTORIES_PER_TASK
    total_target = n_tasks * k_per

    _log(
        f"\n=== Phase 1: live sandbox v2 ({n_tasks} tasks x {k_per} seeds, "
        f"target {total_target} trajectories) ==="
    )

    done: dict[str, dict] = {}
    if TRAJ_PATH.exists():
        try:
            for t in json.loads(TRAJ_PATH.read_text(encoding="utf-8")):
                if _activation_exists(t):
                    done[t["id"]] = t
            if done:
                _log(f"Resuming: {len(done)} trajectories already on disk.")
        except json.JSONDecodeError:
            _log("Warning: corrupt trajectories.json — starting fresh.")

    trajectories = list(done.values())
    if len(trajectories) < total_target:
        collected = collect_trajectories(
            model,
            n_tasks=n_tasks,
            k_per_task=k_per,
            smoke_n=smoke_n,
            existing=trajectories,
        )
        trajectories = collected
        TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    n_success = sum(t["success"] for t in trajectories)
    n_tasks_done = len({t["instance_id"] for t in trajectories})
    _log(
        f"\nPhase 1 done: {len(trajectories)} trajectories across "
        f"{n_tasks_done} tasks, success={n_success}"
    )
    return trajectories


def phase1_collect_replay(model) -> list[dict]:
    from agent.swebench_loader import (
        load_swebench_trajectories,
        replay_and_cache_activations,
    )

    _log("\n=== Phase 1: SWE-bench replay (legacy) ===")
    raw = load_swebench_trajectories()

    done: dict[str, dict] = {}
    if TRAJ_PATH.exists():
        try:
            for t in json.loads(TRAJ_PATH.read_text(encoding="utf-8")):
                if _activation_exists(t):
                    done[t["id"]] = t
            if done:
                _log(f"Resuming: {len(done)} trajectories already on disk.")
        except json.JSONDecodeError:
            _log("Warning: corrupt trajectories.json — starting fresh.")

    trajectories: list[dict] = []
    for i, rec in enumerate(raw):
        tid = rec["id"]
        if tid in done:
            trajectories.append(done[tid])
            _log(f"  [{i+1}/{len(raw)}] skip (already saved) id={tid[:24]}")
            continue

        _log(f"  [{i+1}/{len(raw)}] replay id={tid[:24]} "
             f"(n_steps={rec['n_steps']}, success={rec['success']}) ...")
        traj = replay_and_cache_activations(model, rec)
        if traj is None:
            continue
        trajectories.append(traj)
        TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    n_success = sum(t["success"] for t in trajectories)
    _log(f"\nPhase 1 done: {len(trajectories)} trajectories, "
         f"success={n_success}/{len(trajectories)}")
    return trajectories


def phase1_validate(trajectories: list[dict]) -> list[dict]:
    smoke = os.environ.get("VERITAS_SMOKE_N") is not None
    _log("\n=== Phase 1b: dataset validation (mixed-task filter) ===")
    validated = validate_dataset(trajectories, smoke=smoke)
    TRAJ_PATH.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    _log(f"Validated dataset: {len(validated)} trajectories written to {TRAJ_PATH}")
    return validated


def main() -> None:
    _log("Latent Failure Forecasting PoC pipeline v2 — single process, incremental checkpoints.")
    _log(f"Trajectory source: {TRAJECTORY_SOURCE}")
    _log("Do not start another cell/script while this runs (Colab auto-sends ^C).")

    log_device_choice()
    model = load_model()
    _log(
        f"Model: {MODEL_NAME} on {model.cfg.device} "
        f"({model.cfg.n_layers} layers, d_model={model.cfg.d_model}, n_ctx={model.cfg.n_ctx})"
    )

    try:
        if TRAJECTORY_SOURCE == "replay":
            trajectories = phase1_collect_replay(model)
        else:
            trajectories = phase1_collect_live(model)
        phase1_validate(trajectories)

        _log("\n=== Phase 2: probe (within-task + LOTO by layer x position) ===")
        probe_main()
        _log("\n=== Phase 2b: skepticism checks (v2) ===")
        skepticism_main()
        _log("\n=== Phase 3: visualize ===")
        visualize_main()
        _log("\n=== Phase 4: summary ===")
        summarize_main()
        _log("\n=== Phase 5: run auto-scorer ===")
        score_main()
        _log("\n=== Done ===")
        _log(f"Read: {RESULTS_DIR / 'poc_summary.md'}")
        _log(f"Read: {RESULTS_DIR / 'run_score.md'}")
    except KeyboardInterrupt:
        _log(
            "\nInterrupted. If Phase 1 wrote checkpoints, re-run this script — "
            "it will resume from results/trajectories.json."
        )
        raise
    except Exception:
        traceback.print_exc()
        _log(
            "\nFailed. Check the traceback above. If trajectories.json exists "
            "with partial data, re-run to resume."
        )
        raise


if __name__ == "__main__":
    main()
