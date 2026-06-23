"""Phase 3: visualize probe results (v2 — within-task + LOTO primary).

Figures:
  1. within_task_heatmap.png  — layer x bin within-task micro-AUC
  2. loto_heatmap.png         — layer x bin LOTO AUC
  3. global_heatmap.png       — deprecated global AUC (secondary)
  4. per_task_auc_hist.png    — distribution of per-task AUCs (early bin, best layer)
  5. traj_per_task_hist.png   — trajectories per task histogram

Run:  python -m analysis.visualize_probe
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from interp.activation_cache import RESULTS_DIR

PROBE_PATH = RESULTS_DIR / "probe_results.json"
TRAJ_PATH = RESULTS_DIR / "trajectories.json"

WITHIN_HEATMAP = RESULTS_DIR / "within_task_heatmap.png"
LOTO_HEATMAP = RESULTS_DIR / "loto_heatmap.png"
GLOBAL_HEATMAP = RESULTS_DIR / "global_heatmap_deprecated.png"
PER_TASK_HIST = RESULTS_DIR / "per_task_auc_hist.png"
TRAJ_HIST = RESULTS_DIR / "traj_per_task_hist.png"
BY_POS_PATH = RESULTS_DIR / "within_task_by_position.png"


def _metric_grid(payload: dict, key: str) -> np.ndarray:
    n_layers = payload["n_layers"]
    bins = payload["bins"]
    grid = np.full((n_layers, len(bins)), np.nan)
    for r in payload["results"]:
        val = r.get(key)
        if val is not None:
            grid[r["layer"], r["bin_idx"]] = val
    return grid


def _plot_heatmap(grid: np.ndarray, bins: list, n_layers: int, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5, max(4, n_layers * 0.32)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(range(n_layers))
    ax.set_xlabel("Relative position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.axhline(-0.5, color="none")
    for i in range(n_layers):
        for j in range(len(bins)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"Saved {path}", flush=True)


def plot_by_position(payload: dict) -> None:
    within = _metric_grid(payload, "within_task_micro_auc")
    loto = _metric_grid(payload, "loto_auc")
    bins = payload["bins"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(bins))
    wt_mean = np.nanmean(within, axis=0)
    loto_mean = np.nanmean(loto, axis=0)
    ax.plot(x, wt_mean, marker="o", label="within-task micro-AUC")
    ax.plot(x, loto_mean, marker="s", label="LOTO AUC")
    ax.axhline(0.5, color="grey", linestyle="--", label="chance (0.5)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bins)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative position in trajectory")
    ax.set_ylabel("AUC")
    ax.set_title("v2 primary metrics by relative position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BY_POS_PATH, dpi=130)
    plt.close(fig)
    print(f"Saved {BY_POS_PATH}", flush=True)


def plot_traj_per_task() -> None:
    if not TRAJ_PATH.exists():
        return
    trajs = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))
    counts = Counter(t.get("instance_id", t["id"]) for t in trajs)
    vals = list(counts.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=max(1, len(set(vals))), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Trajectories per task")
    ax.set_ylabel("Number of tasks")
    ax.set_title(f"Trajectory count distribution ({len(counts)} tasks)")
    fig.tight_layout()
    fig.savefig(TRAJ_HIST, dpi=130)
    plt.close(fig)
    print(f"Saved {TRAJ_HIST}", flush=True)


def plot_per_task_auc(payload: dict) -> None:
    early = [r for r in payload["results"] if r["bin_idx"] == 0]
    if not early:
        return
    best = max(
        (r for r in early if r.get("within_task_micro_auc") is not None),
        key=lambda r: r["within_task_micro_auc"],
        default=None,
    )
    if best is None:
        return
    # Per-task AUCs stored in probe run — reconstruct from within_task if available
    # Use macro distribution proxy: read from skepticism if exists
    sk_path = RESULTS_DIR / "skepticism_report.json"
    per_task = []
    if sk_path.exists():
        sk = json.loads(sk_path.read_text(encoding="utf-8"))
        act = sk.get("checks", {}).get("activation_within_task", {})
        per_task = act.get("per_task_aucs") or []

    if not per_task:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(per_task, bins=min(20, len(per_task)), edgecolor="black", alpha=0.7)
    ax.axvline(0.5, color="grey", linestyle="--", label="chance")
    ax.axvline(np.median(per_task), color="red", linestyle="-", label=f"median={np.median(per_task):.2f}")
    ax.set_xlabel("Per-task within-task AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-task AUC distribution (early bin, layer {best['layer']})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PER_TASK_HIST, dpi=130)
    plt.close(fig)
    print(f"Saved {PER_TASK_HIST}", flush=True)


def main() -> None:
    payload = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    bins = payload["bins"]
    n_layers = payload["n_layers"]

    within = _metric_grid(payload, "within_task_micro_auc")
    loto = _metric_grid(payload, "loto_auc")
    global_g = _metric_grid(payload, "global_auc")

    _plot_heatmap(within, bins, n_layers, "Within-task micro-AUC (PRIMARY)", WITHIN_HEATMAP)
    _plot_heatmap(loto, bins, n_layers, "LOTO AUC", LOTO_HEATMAP)
    _plot_heatmap(global_g, bins, n_layers, "Global AUC (deprecated)", GLOBAL_HEATMAP)
    plot_by_position(payload)
    plot_traj_per_task()
    plot_per_task_auc(payload)


if __name__ == "__main__":
    main()
