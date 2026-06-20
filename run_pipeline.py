"""Run the full Veritas PoC pipeline in a single Python process.

Why this exists: on Colab, starting a new cell while another is still running
sends an automatic interrupt (^C) to the old process. The old notebook also
re-loaded the model in every phase script, which looked like a hang after
"Loading weights" and invited accidental re-runs.

This script loads the model once, saves trajectories incrementally (so a crash
mid-run does not lose everything), and runs phases 2-4 in the same process.

Usage (Colab or local):
    python run_pipeline.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runner import TRAJ_PATH, ensure_dataset, run_trajectory  # noqa: E402
from agent.trace_logger import log_trajectory  # noqa: E402
from analysis.correlate import main as correlate_main  # noqa: E402
from analysis.summarize import main as summarize_main  # noqa: E402
from analysis.visualize import main as visualize_main  # noqa: E402
from interp.activation_cache import LAYER, RESULTS_DIR, load_model, log_device_choice  # noqa: E402
from interp.attribution_patch import attribution_patch_scores  # noqa: E402
from interp.attribution_patch import OUT_PATH as FAST_PATH  # noqa: E402
from interp.ground_truth_patch import ground_truth_patch_scores  # noqa: E402
from interp.ground_truth_patch import OUT_PATH as SLOW_PATH  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def phase1_collect(model) -> list[dict]:
    _log("\n=== Phase 1: collect trajectories ===")
    examples = ensure_dataset()
    _log(f"HotpotQA examples: {len(examples)}")

    # Resume if a previous run was interrupted partway through.
    done: dict[str, dict] = {}
    if TRAJ_PATH.exists():
        try:
            for t in json.loads(TRAJ_PATH.read_text(encoding="utf-8")):
                done[t["id"]] = t
            _log(f"Resuming: {len(done)}/{len(examples)} trajectories already on disk.")
        except json.JSONDecodeError:
            _log("Warning: corrupt trajectories.json — starting fresh.")

    trajectories: list[dict] = []
    for i, ex in enumerate(examples):
        tid = str(ex.get("id", ""))
        if tid in done:
            trajectories.append(done[tid])
            _log(f"  [{i+1}/{len(examples)}] skip (already saved) id={tid[:8]}")
            continue

        _log(f"  [{i+1}/{len(examples)}] generating id={tid[:8]} ...")
        traj = run_trajectory(model, ex)
        _log(f"  [{i+1}/{len(examples)}] caching activations ...")
        log_trajectory(model, traj)
        trajectories.append(traj)

        # Checkpoint after every trajectory so Colab interrupts are recoverable.
        TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
        _log(
            f"  [{i+1}/{len(examples)}] saved checkpoint | steps={len(traj['step_positions'])} "
            f"success={traj['success']} answer={traj['generated_answer']!r}"
        )

    n_success = sum(t["success"] for t in trajectories)
    _log(f"\nPhase 1 done: {len(trajectories)} trajectories, success={n_success}/{len(trajectories)}")
    return trajectories


def phase2_fast(model, trajectories: list[dict]) -> dict:
    _log("\n=== Phase 2: attribution patching (fast) ===")
    results = {}
    for i, traj in enumerate(trajectories):
        _log(f"  [{i+1}/{len(trajectories)}] scoring {traj['id'][:8]} ...")
        results[traj["id"]] = attribution_patch_scores(model, traj, LAYER)
    FAST_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"Saved {FAST_PATH}")
    return results


def phase3_slow(model, trajectories: list[dict]) -> dict:
    _log("\n=== Phase 3: activation patching (slow) ===")
    results = {}
    for i, traj in enumerate(trajectories):
        _log(f"  [{i+1}/{len(trajectories)}] scoring {traj['id'][:8]} ...")
        results[traj["id"]] = ground_truth_patch_scores(model, traj, LAYER)
    SLOW_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _log(f"Saved {SLOW_PATH}")
    return results


def main() -> None:
    _log("Veritas PoC pipeline — single process, incremental checkpoints.")
    _log("Do not start another cell/script while this runs (Colab auto-sends ^C).")

    log_device_choice()
    model = load_model()
    _log(
        f"Model: {model.cfg.model_name} on {model.cfg.device} "
        f"({model.cfg.n_layers} layers)"
    )

    try:
        trajectories = phase1_collect(model)
        phase2_fast(model, trajectories)
        phase3_slow(model, trajectories)
        _log("\n=== Phase 3: correlation ===")
        correlate_main()
        _log("\n=== Phase 4: visualize + summary ===")
        visualize_main()
        summarize_main()
        _log("\n=== Done ===")
        _log(f"Read: {RESULTS_DIR / 'poc_summary.md'}")
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
