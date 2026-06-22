"""Run the full Latent Failure Forecasting PoC pipeline in one process.

Why this exists: on Colab, starting a new cell while another is still running
sends an automatic interrupt (^C) to the old process, and re-loading the model
per phase looked like a hang. This script loads the model once, saves
trajectories incrementally (so a crash mid-run does not lose everything), then
runs the probe analysis (which needs no GPU).

Pipeline: collect SWE-bench trajectories + cache step-boundary activations ->
probe (logistic regression by layer x relative-position bin) -> visualize ->
summarize.

Trajectory source: pre-generated SWE-bench agent runs replayed through
Llama-3.2-1B (agent/swebench_loader.py). The HotpotQA loop (agent/runner.py) is
kept as legacy but is no longer wired in — its trajectories were too short
(~3 steps) to test the early->late forecasting hypothesis.

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

from agent.swebench_loader import (  # noqa: E402
    TRAJ_PATH,
    load_swebench_trajectories,
    replay_and_cache_activations,
)
from analysis.probe import main as probe_main  # noqa: E402
from analysis.summarize import main as summarize_main  # noqa: E402
from analysis.visualize_probe import main as visualize_main  # noqa: E402
from interp.activation_cache import RESULTS_DIR, load_model, log_device_choice  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def phase1_collect(model) -> list[dict]:
    _log("\n=== Phase 1: collect SWE-bench trajectories (replay + cache) ===")
    raw = load_swebench_trajectories()
    _log(f"Candidate trajectories: {len(raw)}")

    # Resume if a previous run was interrupted partway through: a trajectory is
    # "done" only if its compact activation .pt actually exists on disk.
    done: dict[str, dict] = {}
    if TRAJ_PATH.exists():
        try:
            for t in json.loads(TRAJ_PATH.read_text(encoding="utf-8")):
                done[t["id"]] = t
            _log(f"Resuming: {len(done)} trajectories already on disk.")
        except json.JSONDecodeError:
            _log("Warning: corrupt trajectories.json — starting fresh.")

    trajectories: list[dict] = []
    for i, rec in enumerate(raw):
        tid = rec["id"]
        if tid in done and "activation_path" in done[tid] \
                and Path(done[tid]["activation_path"]).exists():
            trajectories.append(done[tid])
            _log(f"  [{i+1}/{len(raw)}] skip (already saved) id={tid[:24]}")
            continue

        _log(f"  [{i+1}/{len(raw)}] replay id={tid[:24]} "
             f"(n_steps={rec['n_steps']}, success={rec['success']}) ...")
        traj = replay_and_cache_activations(model, rec)
        if traj is None:  # dropped: over the token budget after truncation
            continue
        trajectories.append(traj)

        # Checkpoint after every trajectory so Colab interrupts are recoverable.
        TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
        _log(
            f"  [{i+1}/{len(raw)}] saved checkpoint | steps={traj['n_steps']} "
            f"seq_len={traj['seq_len']} success={traj['success']}"
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
