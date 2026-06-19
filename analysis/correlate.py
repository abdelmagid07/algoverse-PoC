"""Phase 3: correlate fast vs. slow scores -> the go/no-go number.

Pools per-step scores across all trajectories (aligned by trajectory id and
step order) and computes the Pearson correlation. This r is the actual
decision metric for the PoC.

Run:  python -m analysis.correlate
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import pearsonr

from interp.activation_cache import RESULTS_DIR

FAST_PATH = RESULTS_DIR / "fast_scores.json"
SLOW_PATH = RESULTS_DIR / "slow_scores.json"
OUT_PATH = RESULTS_DIR / "correlation.json"


def interpret(r: float) -> tuple[str, str]:
    if r > 0.6:
        return ("strong", "Green light - bring to team meeting.")
    if r > 0.3:
        return ("moderate", "Cautious green light - flag noise as a known risk.")
    return ("weak", "Do not pursue as primary direction - pivot to another idea.")


def main() -> None:
    with open(FAST_PATH, "r", encoding="utf-8") as f:
        fast = json.load(f)
    with open(SLOW_PATH, "r", encoding="utf-8") as f:
        slow = json.load(f)

    fast_scores, slow_scores = [], []
    for traj_id in fast:
        if traj_id not in slow:
            continue
        f_list, s_list = fast[traj_id], slow[traj_id]
        n = min(len(f_list), len(s_list))
        fast_scores.extend(f_list[:n])
        slow_scores.extend(s_list[:n])

    fast_arr = np.array(fast_scores, dtype=float)
    slow_arr = np.array(slow_scores, dtype=float)

    r, p_value = pearsonr(fast_arr, slow_arr)
    band, action = interpret(r)

    result = {
        "pearson_r": float(r),
        "p_value": float(p_value),
        "n_steps": int(len(fast_scores)),
        "n_trajectories": len(fast),
        "band": band,
        "action": action,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Pearson correlation: r={r:.3f}, p={p_value:.4f}  "
          f"(n={len(fast_scores)} steps across {len(fast)} trajectories)")
    print(f"Band: {band.upper()}  ->  {action}")


if __name__ == "__main__":
    main()
