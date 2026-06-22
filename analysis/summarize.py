"""Phase 4: auto-generate results/poc_summary.md from the probe results.

Reproducible (so it regenerates identically on Colab) and honest: it reports the
class balance, the majority-class chance baseline, the cross-validation spread
(not just means), how many layers show the early->late increase, and the
high-dimensional / low-N regularization caveat.

Decision rule (from NEW_PROPOSAL_POC_GUIDE.md), applied to relative-position bins:
  * STRONG     - accuracy clearly above chance AND clearly increasing early->late
                 on a meaningful number of layers -> green light.
  * CAUTIOUS   - above chance somewhere but roughly flat across position ->
                 pursue, but reframe from "early forecasting" to "internal state
                 correlates with outcome."
  * NO SIGNAL  - near chance everywhere -> do not pursue; fall back to Idea 4.

Run:  python -m analysis.summarize
"""
from __future__ import annotations

import json

import numpy as np

from interp.activation_cache import MODEL_NAME, RESULTS_DIR

PROBE_PATH = RESULTS_DIR / "probe_results.json"
OUT_PATH = RESULTS_DIR / "poc_summary.md"

# Margins, deliberately conservative given the small-N noise.
CHANCE_MARGIN = 0.10   # accuracy must beat chance by this to count as "signal"
INCREASE_MARGIN = 0.10  # late-minus-early delta to count as "increasing"

# Prior HotpotQA run, recorded as labeled constants for the side-by-side
# comparison (C.3). HotpotQA trajectories were ~3 steps, too short to test the
# early->late forecasting hypothesis; this is the baseline the SWE-bench
# migration is meant to beat. Source: the HotpotQA probe summary.
HOTPOTQA_BASELINE = {
    "domain": "HotpotQA (~3 steps)",
    "chance": 0.600,
    "delta": -0.081,           # early->late delta (mean over layers)
    "increasing_layers": 0,
    "n_layers": 16,
    "band": "NO SIGNAL",
}


def _bin_layer_arrays(payload: dict):
    """Return dict bin_idx -> np.array of per-layer acc_mean (NaN where missing)."""
    n_layers = payload["n_layers"]
    n_bins = len(payload["bins"])
    grid = np.full((n_layers, n_bins), np.nan)
    std = np.full((n_layers, n_bins), np.nan)
    auc = np.full((n_layers, n_bins), np.nan)
    for r in payload["results"]:
        if r["acc_mean"] is not None:
            grid[r["layer"], r["bin_idx"]] = r["acc_mean"]
            std[r["layer"], r["bin_idx"]] = r["acc_std"]
        if r["auc"] is not None:
            auc[r["layer"], r["bin_idx"]] = r["auc"]
    return grid, std, auc


def classify(payload: dict) -> dict:
    grid, std, auc = _bin_layer_arrays(payload)
    bins = payload["bins"]
    chance = payload["overall_chance"]
    n_layers = payload["n_layers"]

    per_bin_mean = np.nanmean(grid, axis=0)   # mean over layers
    per_bin_spread = np.nanstd(grid, axis=0)  # spread over layers
    per_bin_auc = np.nanmean(auc, axis=0)

    early_i, late_i = 0, len(bins) - 1
    early_mean = per_bin_mean[early_i]
    late_mean = per_bin_mean[late_i]
    delta = late_mean - early_mean

    # Per-layer early->late increase that also clears chance at the late bin.
    layer_delta = grid[:, late_i] - grid[:, early_i]
    increasing_layers = int(np.nansum(
        (layer_delta > INCREASE_MARGIN) & (grid[:, late_i] > chance + CHANCE_MARGIN)
    ))

    above_chance_anywhere = np.nanmax(per_bin_mean) > chance + CHANCE_MARGIN
    increasing = (delta > INCREASE_MARGIN) and (late_mean > chance + CHANCE_MARGIN) \
        and (increasing_layers >= max(1, round(0.25 * n_layers)))

    if not above_chance_anywhere:
        band = "no_signal"
        action = (
            "Accuracy stays near chance at every relative position and layer. No "
            "usable signal at this scale - do not pursue as the primary direction; "
            "fall back to Idea 4 (goal drift)."
        )
    elif increasing:
        band = "strong"
        action = (
            "Outcome decodability is clearly above chance and increases from the "
            "early to the late portion of trajectories on multiple layers. Strong "
            "signal at small scale - green light to pursue as the primary direction."
        )
    else:
        band = "cautious"
        action = (
            "Outcome decodability is above chance but does not clearly increase "
            "toward the end of trajectories. Some signal exists, but the "
            "'forecasting before the outcome is visible' story is weaker than hoped "
            "- pursue with caution and reframe around 'internal state correlates "
            "with outcome' rather than 'early forecasting.'"
        )

    return {
        "band": band,
        "action": action,
        "chance": chance,
        "per_bin_mean": per_bin_mean,
        "per_bin_spread": per_bin_spread,
        "per_bin_auc": per_bin_auc,
        "delta": delta,
        "increasing_layers": increasing_layers,
        "grid": grid,
        "std": std,
    }


def main() -> None:
    payload = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    bins = payload["bins"]
    n = payload["n_trajectories"]
    n_success = payload["n_success"]
    n_fail = payload["n_fail"]
    chance = payload["overall_chance"]

    c = classify(payload)
    grid, std = c["grid"], c["std"]

    # Best layer at the late bin (for an honest "spread, not just mean" callout).
    late_i = len(bins) - 1
    late_col = grid[:, late_i]
    if np.isfinite(late_col).any():
        best_layer = int(np.nanargmax(late_col))
        best_late = late_col[best_layer]
        best_late_std = std[best_layer, late_i]
        best_str = (f"layer {best_layer}: {best_late:.3f} +/- {best_late_std:.3f} "
                    f"(fold spread)")
    else:
        best_str = "n/a (late bin not probeable)"

    # Per-bin table rows.
    bin_rows = []
    for j, name in enumerate(bins):
        m = c["per_bin_mean"][j]
        sp = c["per_bin_spread"][j]
        au = c["per_bin_auc"][j]
        cells = [r for r in payload["results"] if r["bin_idx"] == j]
        ns = [r["n"] for r in cells]
        nrow = ns[0] if ns else 0
        npos = cells[0]["n_pos"] if cells else 0
        nneg = cells[0]["n_neg"] if cells else 0
        m_str = "n/a" if np.isnan(m) else f"{m:.3f}"
        sp_str = "n/a" if np.isnan(sp) else f"{sp:.3f}"
        au_str = "n/a" if np.isnan(au) else f"{au:.3f}"
        bin_rows.append(
            f"| {name} | {m_str} | {sp_str} | {au_str} | {nrow} ({npos}+/{nneg}-) |"
        )

    band_label = {"strong": "STRONG", "cautious": "CAUTIOUS", "no_signal": "NO SIGNAL"}[c["band"]]

    # --- HotpotQA vs. SWE-bench side-by-side (C.3) + interpretation (C.4) ----
    hb = HOTPOTQA_BASELINE
    comparison_rows = "\n".join([
        "| Metric | HotpotQA (prior) | SWE-bench Lite (this run) |",
        "|---|---|---|",
        f"| Domain / length | {hb['domain']} | SWE-bench Lite (8-20 steps) |",
        f"| Chance baseline | {hb['chance']:.3f} | {chance:.3f} |",
        f"| Early->late delta (mean over layers) | {hb['delta']:+.3f} | {c['delta']:+.3f} |",
        f"| Layers showing the increase | {hb['increasing_layers']}/{hb['n_layers']} | "
        f"{c['increasing_layers']}/{payload['n_layers']} |",
        f"| Decision band | {hb['band']} | {band_label} |",
    ])
    if c["band"] == "no_signal":
        interpretation = (
            "Still null at proper trajectory length. The prior HotpotQA null was "
            "therefore **not** purely a length artifact - on longer SWE-bench "
            "trajectories the outcome is still not linearly decodable from the "
            "residual stream at this scale. This is a more trustworthy null."
        )
    else:
        interpretation = (
            "Signal appears once trajectories are long enough. This is consistent "
            "with the prior HotpotQA null being a **length artifact**: ~3-step "
            "trajectories were too short for an early->late forecasting trend to "
            "exist, while SWE-bench's longer horizon makes it measurable."
        )

    imbalance_note = ""
    if min(n_success, n_fail) < 0.3 * n:
        imbalance_note = (
            f" Class balance is skewed ({n_success} success / {n_fail} fail), so "
            "accuracy is read against the majority-class chance baseline above, and "
            "AUC is the more trustworthy metric here."
        )

    md = f"""# Latent Failure Forecasting PoC - Results Summary

_Auto-generated by `analysis/summarize.py` from `results/probe_results.json`._

## Verdict

| Metric | Value |
|---|---|
| Model | `{MODEL_NAME}` |
| Trajectories | {n} (success={n_success}, fail={n_fail}) |
| Chance baseline (majority class) | {chance:.3f} |
| Layers probed | {payload['n_layers']} (d_model={payload['d_model']}) |
| Early->late delta (mean over layers) | {c['delta']:+.3f} |
| Layers showing the increase | {c['increasing_layers']}/{payload['n_layers']} |
| Best late-bin probe | {best_str} |
| **Decision band** | **{band_label}** |

## Accuracy by relative position (mean over layers)

| Relative position | Acc (mean over layers) | Spread across layers | AUC (OOF) | n (pos/neg) |
|---|---|---|---|---|
{chr(10).join(bin_rows)}

Chance baseline = {chance:.3f}. "Spread across layers" is the std of the per-layer
accuracies; the per-cell cross-validation fold spread is in `probe_results.json`
(`acc_std`). See `accuracy_by_position.png`, `accuracy_by_layer.png`, and
`probe_heatmap.png`.

## HotpotQA vs. SWE-bench (length-artifact check)

The PoC moved from HotpotQA to SWE-bench Lite specifically because HotpotQA
trajectories were too short (~3 steps) to test "decodability grows as the agent
nears its own answer." Same probe methodology, longer-horizon data:

{comparison_rows}

**Interpretation.** {interpretation}

## Class balance

{n_success} successes / {n_fail} failures out of {n} trajectories.{imbalance_note}

## Recommendation

{c['action']}

## Methodological caveats (read before trusting the numbers)

1. **Small N / cross-validation noise.** With ~{n} trajectories, CV folds hold
   only a handful of examples each, so accuracy is noisy. We report the spread
   across folds (`acc_std`) and across layers, not just the mean - do not over-read
   any single accuracy number.
2. **High dimensions vs. few samples.** The residual stream is {payload['d_model']}-dim
   but N ~ {n}, so the probe is heavily regularized (StandardScaler + L2,
   C={payload['probe_C']}). Absolute accuracy is regularization-sensitive; the
   *shape* of the early->late trend matters more than the absolute level.
3. **Relative-position binning.** Steps are bucketed into early/mid/late thirds of
   each trajectory (not absolute step index) so that short and long trajectories
   both contribute to every bin, removing the "fewer examples at late steps"
   confound. Each trajectory contributes one mean-pooled row per bin.
4. **PoC scope.** No causal validation, no SAE features, single domain
   (SWE-bench Lite), logistic-regression probes only. This is a
   direction-validation PoC, not a publication-grade measurement.

### SWE-bench migration caveats (flagged, not silently fixed)

5. **Step-boundary convention.** One step = one `ai` (assistant) turn; its step
   position is that turn's final token; `user`/`system` turns are observations,
   not steps. This is a real choice - a different convention (e.g. splitting on
   tool calls) could move reasoning across the early/mid/late bins and change the
   trend.
6. **Context truncation.** TransformerLens loads Llama with `n_ctx=2048` (rotary
   embeddings are precomputed only that far — longer sequences crash). Each
   observation is head-truncated (`OBS_TOKEN_CAP`), `ai` turns are capped too
   (`AI_TOKEN_CAP`), and if still over budget the oldest (user, ai) pairs after
   the issue preamble are dropped. Trajectories that cannot fit or fall below
   `MIN_STEPS` after trimming are dropped (and logged), not silently clipped.
7. **Foreign-trajectory replay.** The trajectories were generated by other
   agents/models; we replay their text through `{MODEL_NAME}` and use the
   dataset's own `target` label. So this measures "does {MODEL_NAME}'s internal
   state while *reading* the trajectory encode the eventual outcome" - a proxy
   for the proposal's "agent's own internal forecast," not a literal measurement
   of it.

## Reproduce

```bash
python run_pipeline.py          # collect SWE-bench trajectories + cache, then probe/visualize/summarize
# or, step by step:
python -m agent.swebench_loader # collect + replay + cache step-boundary activations
python -m analysis.probe        # layer x relative-position probes (CV)
python -m analysis.visualize_probe
python -m analysis.summarize
```
"""
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_PATH}", flush=True)
    print(f"Decision band: {band_label}  "
          f"(delta={c['delta']:+.3f}, increasing_layers={c['increasing_layers']})",
          flush=True)


if __name__ == "__main__":
    main()
