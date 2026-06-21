"""Phase 3: visualize the probe results.

Three figures, all with the majority-class chance baseline drawn in:
  1. probe_heatmap.png        - layer x relative-position accuracy grid (the
                                complete, un-cherry-picked picture)
  2. accuracy_by_position.png - accuracy vs relative position (early/mid/late),
                                mean over layers with +/-1 std spread; this is
                                the headline "does signal grow toward the end?"
  3. accuracy_by_layer.png    - accuracy vs layer (one line per bin), the
                                "which layer carries the signal" view

Run:  python -m analysis.visualize_probe
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from interp.activation_cache import RESULTS_DIR

PROBE_PATH = RESULTS_DIR / "probe_results.json"
HEATMAP_PATH = RESULTS_DIR / "probe_heatmap.png"
BY_POS_PATH = RESULTS_DIR / "accuracy_by_position.png"
BY_LAYER_PATH = RESULTS_DIR / "accuracy_by_layer.png"


def _acc_grid(payload: dict) -> np.ndarray:
    """[n_layers x n_bins] accuracy grid; NaN where a cell was not probeable."""
    n_layers = payload["n_layers"]
    bins = payload["bins"]
    grid = np.full((n_layers, len(bins)), np.nan)
    for r in payload["results"]:
        if r["acc_mean"] is not None:
            grid[r["layer"], r["bin_idx"]] = r["acc_mean"]
    return grid


def _std_grid(payload: dict) -> np.ndarray:
    n_layers = payload["n_layers"]
    bins = payload["bins"]
    grid = np.full((n_layers, len(bins)), np.nan)
    for r in payload["results"]:
        if r["acc_std"] is not None:
            grid[r["layer"], r["bin_idx"]] = r["acc_std"]
    return grid


def plot_heatmap(payload: dict) -> None:
    grid = _acc_grid(payload)
    bins = payload["bins"]
    n_layers = payload["n_layers"]

    fig, ax = plt.subplots(figsize=(5, max(4, n_layers * 0.32)))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(range(n_layers))
    ax.set_xlabel("Relative position")
    ax.set_ylabel("Layer")
    ax.set_title(f"Probe accuracy (chance={payload['overall_chance']:.2f})")
    for i in range(n_layers):
        for j in range(len(bins)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    fig.colorbar(im, ax=ax, label="CV accuracy")
    fig.tight_layout()
    fig.savefig(HEATMAP_PATH, dpi=130)
    plt.close(fig)
    print(f"Saved {HEATMAP_PATH}", flush=True)


def plot_by_position(payload: dict) -> None:
    grid = _acc_grid(payload)
    bins = payload["bins"]
    chance = payload["overall_chance"]

    mean = np.nanmean(grid, axis=0)
    std = np.nanstd(grid, axis=0)
    x = range(len(bins))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, mean, yerr=std, marker="o", capsize=4, label="mean over layers")
    ax.axhline(chance, color="grey", linestyle="--", label=f"chance ({chance:.2f})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(bins)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Relative position in trajectory")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Outcome decodability by relative position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BY_POS_PATH, dpi=130)
    plt.close(fig)
    print(f"Saved {BY_POS_PATH}", flush=True)


def plot_by_layer(payload: dict) -> None:
    grid = _acc_grid(payload)
    bins = payload["bins"]
    chance = payload["overall_chance"]
    n_layers = payload["n_layers"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for j, name in enumerate(bins):
        ax.plot(range(n_layers), grid[:, j], marker="o", alpha=0.8, label=name)
    ax.axhline(chance, color="grey", linestyle="--", label=f"chance ({chance:.2f})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Outcome decodability by layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(BY_LAYER_PATH, dpi=130)
    plt.close(fig)
    print(f"Saved {BY_LAYER_PATH}", flush=True)


def main() -> None:
    payload = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    plot_heatmap(payload)
    plot_by_position(payload)
    plot_by_layer(payload)


if __name__ == "__main__":
    main()
