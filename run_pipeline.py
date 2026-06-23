"""Run the full Latent Failure Forecasting PoC pipeline in one process.

Pipeline: collect live sandbox trajectories (default) or legacy SWE-bench replay
-> probe -> skepticism -> visualize -> summarize.

Set VERITAS_TRAJECTORY_SOURCE=replay to use foreign-trajectory replay instead of
the live Llama sandbox agent.

Usage (Colab or local):
    python run_pipeline.py
    VERITAS_SMOKE_N=2 python run_pipeline.py   # quick smoke (2 trajectories)
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

from analysis.probe import main as probe_main  # noqa: E402
from analysis.skepticism import main as skepticism_main  # noqa: E402
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
    from agent.sandbox_runner import collect_trajectories, N_TRAJECTORIES

    smoke = os.environ.get("VERITAS_SMOKE_N")
    smoke_n = int(smoke) if smoke else None
    n = smoke_n or N_TRAJECTORIES

    _log(f"\n=== Phase 1: live sandbox trajectories (target={n}) ===")

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

    if len(done) >= n:
        trajectories = list(done.values())[:n]
        _log(f"Phase 1 skipped: already have {len(trajectories)} trajectories.")
        return trajectories

    trajectories = list(done.values())
    need = n - len(trajectories)
    if need > 0:
        collected = collect_trajectories(
            model, n=n, smoke_n=smoke_n, existing=trajectories
        )
        trajectories = collected[:n]
        TRAJ_PATH.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    n_success = sum(t["success"] for t in trajectories[:n])
    _log(f"\nPhase 1 done: {min(len(trajectories), n)} trajectories, "
         f"success={n_success}")
    return trajectories[:n]


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


def main() -> None:
    _log("Latent Failure Forecasting PoC pipeline — single process, incremental checkpoints.")
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
            phase1_collect_replay(model)
        else:
            phase1_collect_live(model)
        _log("\n=== Phase 2: probe (logistic regression by layer x position) ===")
        probe_main()
        _log("\n=== Phase 2b: skepticism checks ===")
        skepticism_main()
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
