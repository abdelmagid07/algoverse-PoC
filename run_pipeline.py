"""Run the full Latent Failure Forecasting PoC pipeline in one process.

Why this exists: on Colab, starting a new cell while another is still running
sends an automatic interrupt (^C) to the old process, and re-loading the model
per phase looked like a hang. This script loads the model once, saves
trajectories incrementally (so a crash mid-run does not lose everything), then
runs the probe analysis (which needs no GPU).

Pipeline: collect trajectories + cache activations -> probe (logistic
regression by layer x relative-position bin) -> visualize -> summarize.

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
from analysis.probe import main as probe_main  # noqa: E402
from analysis.summarize import main as summarize_main  # noqa: E402
from analysis.visualize_probe import main as visualize_main  # noqa: E402
from interp.activation_cache import RESULTS_DIR, load_model, log_device_choice  # noqa: E402


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
            _log(f"Resuming: {len(done)} trajectories already on disk.")
        except json.JSONDecodeError:
            _log("Warning: corrupt trajectories.json — starting fresh.")

    trajectories: list[dict] = []
    for i, ex in enumerate(examples):
        tid = str(ex.get("id", ""))
        if tid in done and "activation_path" in done[tid] \
                and Path(done[tid]["activation_path"]).exists():
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
    _log(f"\nPhase 1 done: {len(trajectories)} trajectories, "
         f"success={n_success}/{len(trajectories)}")
    return trajectories


def main() -> None:
    _log("Latent Failure Forecasting PoC pipeline — single process, incremental checkpoints.")
    _log("Do not start another cell/script while this runs (Colab auto-sends ^C).")

    log_device_choice()
    model = load_model()
    _log(
        f"Model: {model.cfg.model_name} on {model.cfg.device} "
        f"({model.cfg.n_layers} layers)"
    )

    try:
        phase1_collect(model)
        _log("\n=== Phase 2: probe (logistic regression by layer x position) ===")
        probe_main()
        _log("\n=== Phase 3: visualize ===")
        visualize_main()
        _log("\n=== Phase 4: summary ===")
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
