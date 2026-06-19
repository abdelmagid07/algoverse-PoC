"""Phase 4: visualize step importance by position.

Plots the fast (attribution) importance score by step index, one line per
trajectory overlaid, with successful and failed trajectories styled
differently. Look for whether high importance clusters at consistent positions
(e.g. always the step right before the answer).

Run:  python -m analysis.visualize
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

from interp.activation_cache import RESULTS_DIR

FAST_PATH = RESULTS_DIR / "fast_scores.json"
TRAJ_PATH = RESULTS_DIR / "trajectories.json"
OUT_PATH = RESULTS_DIR / "importance_by_step.png"


def main() -> None:
    with open(FAST_PATH, "r", encoding="utf-8") as f:
        fast = json.load(f)
    with open(TRAJ_PATH, "r", encoding="utf-8") as f:
        trajectories = json.load(f)
    success = {t["id"]: t["success"] for t in trajectories}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    seen_labels = set()
    for traj_id, scores in fast.items():
        ok = success.get(traj_id, False)
        color = "tab:green" if ok else "tab:red"
        label = "success" if ok else "fail"
        x = list(range(1, len(scores) + 1))
        ax.plot(
            x, scores,
            marker="o", color=color, alpha=0.7,
            label=label if label not in seen_labels else None,
        )
        seen_labels.add(label)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Step index")
    ax.set_ylabel("Attribution importance (fast)")
    ax.set_title("Step importance by position (one line per trajectory)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130)
    print(f"Saved plot to {OUT_PATH}")


if __name__ == "__main__":
    main()
