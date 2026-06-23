"""Phase 4: auto-generate results/poc_summary.md from v2 probe results.

Decision rule uses within-task AUC and LOTO AUC (not global accuracy).
Global metrics are reported as deprecated confounded baselines.

Run:  python -m analysis.summarize
"""
from __future__ import annotations

import json

import numpy as np

from interp.activation_cache import MODEL_NAME, RESULTS_DIR

PROBE_PATH = RESULTS_DIR / "probe_results.json"
OUT_PATH = RESULTS_DIR / "poc_summary.md"
SKEPTICISM_JSON = RESULTS_DIR / "skepticism_report.json"
SKEPTICISM_MD = RESULTS_DIR / "skepticism_report.md"

AUC_THRESHOLD = 0.55
SHUFFLE_DROP_MIN = 0.15
TEXT_MARGIN = 0.05


def _trajectory_source() -> str:
    path = RESULTS_DIR / "trajectories.json"
    if not path.exists():
        return "unknown"
    try:
        trajs = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unknown"
    sources = {t.get("source", "replay") for t in trajs}
    if "live_sandbox" in sources:
        return "live_sandbox"
    return "replay"


def _metric_grid(payload: dict, key: str) -> np.ndarray:
    n_layers = payload["n_layers"]
    n_bins = len(payload["bins"])
    grid = np.full((n_layers, n_bins), np.nan)
    for r in payload["results"]:
        val = r.get(key)
        if val is not None:
            grid[r["layer"], r["bin_idx"]] = val
    return grid


def classify_v2(payload: dict, skepticism: dict | None) -> dict:
    within = _metric_grid(payload, "within_task_micro_auc")
    loto = _metric_grid(payload, "loto_auc")
    bins = payload["bins"]

    per_bin_within = np.nanmean(within, axis=0)
    per_bin_loto = np.nanmean(loto, axis=0)

    best_within = float(np.nanmax(within)) if np.isfinite(within).any() else None
    best_loto = float(np.nanmax(loto)) if np.isfinite(loto).any() else None

    early_i, late_i = 0, len(bins) - 1
    wt_delta = per_bin_within[late_i] - per_bin_within[early_i]

    checks = skepticism.get("checks", {}) if skepticism else {}
    shuffle_drop = checks.get("within_instance_shuffle", {}).get("auc_drop")
    wt_act = checks.get("activation_within_task", {}).get("micro_auc")
    text_wt = checks.get("text_baseline_within_task", {}).get("within_task_micro_auc")

    conditions = {
        "within_task_above_chance": best_within is not None and best_within > AUC_THRESHOLD,
        "loto_above_chance": best_loto is not None and best_loto > AUC_THRESHOLD,
        "shuffle_breaks_signal": shuffle_drop is not None and shuffle_drop >= SHUFFLE_DROP_MIN,
        "activation_beats_text": (
            wt_act is not None and text_wt is not None
            and wt_act > text_wt + TEXT_MARGIN
        ),
    }

    n_pass = sum(conditions.values())
    if n_pass >= 3 and conditions["within_task_above_chance"]:
        band = "validated"
        action = (
            "v2 controls largely pass: within-task and/or LOTO AUC above chance, "
            "shuffle ablation and text baseline support internal trajectory signal. "
            "Proceed with scaled-up collection."
        )
    elif conditions["within_task_above_chance"] or conditions["loto_above_chance"]:
        band = "cautious"
        action = (
            "Partial v2 signal: some within-task or LOTO AUC above chance, but not all "
            "validation conditions met. Do not claim latent forecasting without "
            "passing shuffle and text baselines."
        )
    else:
        band = "no_signal"
        action = (
            "No within-task or LOTO signal above threshold. The probe is not decoding "
            "trajectory outcome within tasks — do not pursue without redesign."
        )

    return {
        "band": band,
        "action": action,
        "best_within_auc": best_within,
        "best_loto_auc": best_loto,
        "per_bin_within": per_bin_within,
        "per_bin_loto": per_bin_loto,
        "wt_delta": wt_delta,
        "conditions": conditions,
        "within_grid": within,
        "loto_grid": loto,
    }


def main() -> None:
    payload = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    skepticism = None
    if SKEPTICISM_JSON.exists():
        skepticism = json.loads(SKEPTICISM_JSON.read_text(encoding="utf-8"))

    bins = payload["bins"]
    n = payload["n_trajectories"]
    n_success = payload["n_success"]
    n_fail = payload["n_fail"]

    c = classify_v2(payload, skepticism)

    best_layer_within = int(np.unravel_index(np.nanargmax(c["within_grid"]), c["within_grid"].shape)[0])
    best_wt = c["best_within_auc"]
    best_loto = c["best_loto_auc"]

    bin_rows = []
    for j, name in enumerate(bins):
        wt = c["per_bin_within"][j]
        lo = c["per_bin_loto"][j]
        wt_str = "n/a" if np.isnan(wt) else f"{wt:.3f}"
        lo_str = "n/a" if np.isnan(lo) else f"{lo:.3f}"
        bin_rows.append(f"| {name} | {wt_str} | {lo_str} |")

    band_label = {
        "validated": "VALIDATED (v2)",
        "cautious": "CAUTIOUS (v2)",
        "no_signal": "NO SIGNAL (v2)",
    }[c["band"]]

    cond = c["conditions"]
    cond_rows = "\n".join([
        f"| Within-task AUC > {AUC_THRESHOLD} | {'PASS' if cond['within_task_above_chance'] else 'FAIL'} |",
        f"| LOTO AUC > {AUC_THRESHOLD} | {'PASS' if cond['loto_above_chance'] else 'FAIL'} |",
        f"| Shuffle drop >= {SHUFFLE_DROP_MIN} | {'PASS' if cond['shuffle_breaks_signal'] else 'FAIL'} |",
        f"| Activation > text + {TEXT_MARGIN} | {'PASS' if cond['activation_beats_text'] else 'FAIL'} |",
    ])

    source = _trajectory_source()
    is_live = source == "live_sandbox"

    skepticism_block = ""
    if SKEPTICISM_MD.exists():
        skepticism_block = (
            "\n## Skepticism checks\n\n"
            + SKEPTICISM_MD.read_text(encoding="utf-8").split("\n", 1)[-1].strip()
            + "\n"
        )

    md = f"""# Latent Failure Forecasting PoC v2 — Results Summary

_Auto-generated by `analysis/summarize.py` from `results/probe_results.json`._

## Verdict (v2 primary metrics)

| Metric | Value |
|---|---|
| Model | `{MODEL_NAME}` |
| Trajectories (mixed tasks) | {n} (success={n_success}, fail={n_fail}) |
| Best within-task micro-AUC | {best_wt if best_wt is not None else 'n/a'} (layer {best_layer_within}) |
| Best LOTO AUC | {best_loto if best_loto is not None else 'n/a'} |
| Early→late within-task delta | {c['wt_delta']:+.3f} (supplementary only) |
| **Decision band** | **{band_label}** |

## Validation conditions

| Condition | Result |
|---|---|
{cond_rows}

## Within-task and LOTO AUC by relative position

| Relative position | Within-task micro-AUC | LOTO AUC |
|---|---|---|
{chr(10).join(bin_rows)}

Chance baseline for balanced within-task classification = **0.5**. Do **not** interpret
global accuracy or global AUC (confounded by task identity in v1 design).

See `within_task_heatmap.png`, `loto_heatmap.png`, `within_task_by_position.png`.

## Recommendation

{c['action']}
{skepticism_block}
## Methodological notes (v2)

1. **Multi-seed per task.** Each task has K trajectories (different seeds / sampling).
   Labels vary within task so probes cannot shortcut via task identity alone.
2. **Mixed-task filter.** Tasks with only successes or only failures are dropped.
3. **Primary metrics:** within-task micro-AUC and LOTO AUC. Global stratified CV is
   deprecated (`metric_tier: deprecated_confounded`).
4. **PoC scope.** Logistic probes on residual stream; correlational not causal.

## Reproduce

```bash
python run_pipeline.py
VERITAS_SMOKE_N=2 python run_pipeline.py   # smoke: 2 tasks x 3 seeds
python -m analysis.run_scorer              # score an existing run
```

See also `results/run_score.md` for INVALID / INCONCLUSIVE / MEANINGFUL verdict.
"""
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {OUT_PATH}", flush=True)
    print(f"Decision band: {band_label}  (within={best_wt}, loto={best_loto})", flush=True)


if __name__ == "__main__":
    main()
