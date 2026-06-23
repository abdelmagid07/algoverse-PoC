"""Skepticism checks on probe results (v2 — within-task focused).

Validates whether above-chance decodability survives controls that rule out
task-identity memorization:

  1. Within-task label shuffle (PRIMARY leakage test)
  2. Within-task activation probe vs text baseline
  3. LOTO (leave-one-task-out) generalization
  4. Per-task permutation null
  5. Bootstrap CIs on trajectory resampling (unique traj IDs)

Run:  python -m analysis.skepticism
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from analysis.probe import (
    BIN_NAMES,
    PROBE_C,
    RANDOM_STATE,
    build_dataset,
    loto_cv,
    probe_one,
    relative_bin,
    within_task_cv,
)
from interp.activation_cache import RESULTS_DIR

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
PROBE_PATH = RESULTS_DIR / "probe_results.json"
OUT_PATH = RESULTS_DIR / "skepticism_report.json"
OUT_MD = RESULTS_DIR / "skepticism_report.md"

EARLY_BIN = 0
N_PERM = 300
N_BOOT = 400
MAX_TFIDF_FEATURES = 256
_STRIP_PATTERNS = [
    re.compile(r"fix_[a-z0-9_]+", re.I),
    re.compile(r"solution\.py", re.I),
    re.compile(r"\b[a-z_]+\.py\b", re.I),
]


def _strip_instance_cues(text: str, traj: dict) -> str:
    out = text
    inst = str(traj.get("instance_id", ""))
    if inst:
        out = out.replace(inst, " ")
    for pat in _STRIP_PATTERNS:
        out = pat.sub(" ", out)
    return " ".join(out.split())


def _early_text(traj: dict) -> str:
    texts = traj.get("step_texts") or []
    total = len(traj.get("step_positions") or texts)
    if total == 0:
        return ""
    parts = []
    for s, text in enumerate(texts[:total]):
        if relative_bin(s, total) == EARLY_BIN and text:
            parts.append(text)
    return " ".join(parts)


def build_indexed_rows(trajectories: list[dict]) -> tuple[dict, dict, dict, dict, int, int]:
    features, labels, meta, n_layers, d_model = build_dataset(trajectories)
    traj_by_id = {t["id"]: t for t in trajectories}
    rows: dict[int, list[dict]] = {b: [] for b in range(len(BIN_NAMES))}

    for b in range(len(BIN_NAMES)):
        for m in meta[b]:
            traj = traj_by_id.get(m["traj_id"], {})
            early = _early_text(traj) if traj else ""
            rows[b].append({
                "traj_id": m["traj_id"],
                "instance_id": m["instance_id"],
                "success": int(bool(traj.get("success"))) if traj else 0,
                "early_text": early,
                "early_text_stripped": _strip_instance_cues(early, traj) if traj else "",
            })

    return rows, features, labels, meta, n_layers, d_model


def best_layer_early(probe_payload: dict | None, features, labels, meta, n_layers: int) -> int:
    if probe_payload:
        early = [
            r for r in probe_payload["results"]
            if r["bin_idx"] == EARLY_BIN and r.get("within_task_micro_auc") is not None
        ]
        if early:
            return int(max(early, key=lambda r: r["within_task_micro_auc"])["layer"])
    best_l, best_auc = 0, -1.0
    groups = np.array([m["instance_id"] for m in meta[EARLY_BIN]])
    for layer in range(n_layers):
        X = np.array(features[(layer, EARLY_BIN)], dtype=np.float64)
        y = np.array(labels[EARLY_BIN], dtype=int)
        cell = within_task_cv(X, y, groups)
        auc = cell.get("micro_auc")
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_l = layer
    return best_l


def shuffle_within_instance(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y2 = y.copy()
    for inst in np.unique(groups):
        idx = np.where(groups == inst)[0]
        if len(idx) > 1:
            y2[idx] = rng.permutation(y[idx])
    return y2


def within_task_permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
) -> dict:
    """Null distribution of within-task AUC under per-task label shuffles."""
    real = within_task_cv(X, y, groups)
    real_auc = real.get("micro_auc")
    if real_auc is None:
        return {"real_auc": None, "null_mean": None, "p_value": None, "note": real.get("note", "")}

    null_aucs = []
    for _ in range(n_perm):
        y_perm = shuffle_within_instance(y, groups, rng)
        cell = within_task_cv(X, y_perm, groups)
        if cell.get("micro_auc") is not None:
            null_aucs.append(cell["micro_auc"])
    null_aucs_arr = np.array(null_aucs)
    p = float((null_aucs_arr >= real_auc).mean()) if len(null_aucs_arr) else None
    return {
        "real_auc": float(real_auc),
        "null_mean": float(null_aucs_arr.mean()) if len(null_aucs_arr) else None,
        "null_std": float(null_aucs_arr.std()) if len(null_aucs_arr) else None,
        "p_value": p,
        "n_perm": len(null_aucs_arr),
    }


def text_baseline_within_task(texts: list[str], y: np.ndarray, groups: np.ndarray) -> dict:
    if len(texts) != len(y) or len(y) < 4:
        return {"note": "insufficient rows for text baseline"}
    vec = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, stop_words="english")
    try:
        X = vec.fit_transform(texts).toarray()
    except ValueError as exc:
        return {"note": f"tfidf failed: {exc}"}
    wt = within_task_cv(X, y, groups)
    loto = loto_cv(X, y, groups)
    return {
        "within_task_micro_auc": wt.get("micro_auc"),
        "within_task_macro_auc": wt.get("macro_auc"),
        "loto_auc": loto.get("auc"),
        "note": wt.get("note", ""),
    }


def bootstrap_within_task_ci(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    traj_ids: list[str],
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> dict:
    """Bootstrap unique trajectory IDs, then within-task AUC."""
    n = len(y)
    if n < 4:
        return {"note": "too few rows for bootstrap"}
    unique_ids = list(dict.fromkeys(traj_ids))
    aucs = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        idx = [i for i, tid in enumerate(traj_ids) if tid in set(sampled)]
        # Include all rows for sampled trajectory IDs (with replacement weight)
        idx = []
        for tid in sampled:
            idx.extend([i for i, t in enumerate(traj_ids) if t == tid])
        if len(idx) < 4:
            continue
        idx_arr = np.array(idx)
        if len(np.unique(y[idx_arr])) < 2:
            continue
        cell = within_task_cv(X[idx_arr], y[idx_arr], groups[idx_arr])
        if cell.get("micro_auc") is not None:
            aucs.append(cell["micro_auc"])
    if not aucs:
        return {"note": "no valid bootstrap samples"}
    aucs_arr = np.array(aucs)
    lo = float(np.percentile(aucs_arr, (1 - ci) / 2 * 100))
    hi = float(np.percentile(aucs_arr, (1 + ci) / 2 * 100))
    return {
        "auc_mean": float(aucs_arr.mean()),
        "auc_std": float(aucs_arr.std()),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot_effective": int(len(aucs_arr)),
    }


def _verdict(checks: dict) -> str:
    flags = []
    wt = checks.get("activation_within_task", {})
    loto = checks.get("loto", {})
    perm = checks.get("within_task_permutation", {})
    shuffle = checks.get("within_instance_shuffle", {})
    text = checks.get("text_baseline_within_task", {})

    wt_auc = wt.get("micro_auc")
    loto_auc = loto.get("auc")

    if wt_auc is not None and wt_auc < 0.55:
        flags.append("within-task AUC weak (<0.55)")
    if loto_auc is not None and loto_auc < 0.55:
        flags.append("LOTO AUC weak (<0.55)")

    drop = shuffle.get("auc_drop")
    if drop is not None and drop < 0.15:
        flags.append(
            "within-task label shuffle barely hurts AUC (task-identity leakage likely)"
        )

    if perm.get("p_value") is not None and perm["p_value"] > 0.05:
        flags.append("within-task permutation not significant (p>0.05)")

    text_auc = text.get("within_task_micro_auc")
    if text_auc is not None and wt_auc is not None:
        if text_auc >= wt_auc - 0.03:
            flags.append("early text alone nearly matches activations (within-task)")
        elif wt_auc - text_auc >= 0.05:
            flags.append("activations beat text baseline within-task")

    if not flags and wt_auc is not None and wt_auc > 0.55:
        return (
            "v2 controls favorable: within-task AUC above chance, LOTO generalizes, "
            "per-task shuffle collapses or permutation significant."
        )
    if not flags:
        return "Mixed results — inspect per-task AUC distribution and ablations."
    return "Caveats: " + "; ".join(flags) + "."


def render_markdown(report: dict) -> str:
    c = report["checks"]
    lines = [
        "# Skepticism checks v2 (auto-generated)",
        "",
        f"Reference: early bin, layer {report['reference_layer']}, "
        f"n={report['n_trajectories']} trajectories, "
        f"{report['n_instances']} distinct tasks, "
        f"{report['n_mixed_tasks']} with mixed labels.",
        "",
        "## Results",
        "",
        "| Check | Metric | Value |",
        "|---|---|---|",
    ]
    act = c["activation_within_task"]
    lines.append(
        f"| Activation probe (within-task) | micro-AUC | {act.get('micro_auc', 'n/a')} "
        f"(macro {act.get('macro_auc', 'n/a')}) |"
    )
    loto = c["loto"]
    lines.append(
        f"| LOTO (leave-one-task-out) | AUC | {loto.get('auc', 'n/a')} |"
    )
    txt = c["text_baseline_within_task"]
    lines.append(
        f"| Early text TF-IDF (within-task) | micro-AUC | "
        f"{txt.get('within_task_micro_auc', 'n/a')} |"
    )
    txt_s = c.get("text_baseline_stripped", {})
    lines.append(
        f"| Early text TF-IDF stripped (within-task) | micro-AUC | "
        f"{txt_s.get('within_task_micro_auc', 'n/a')} |"
    )
    perm = c["within_task_permutation"]
    lines.append(
        f"| Within-task label shuffle null | p-value | {perm.get('p_value', 'n/a')} "
        f"(null mean {perm.get('null_mean', 'n/a')}) |"
    )
    wis = c["within_instance_shuffle"]
    drop_s = f"{wis['auc_drop']:+.3f}" if wis.get("auc_drop") is not None else "n/a"
    lines.append(
        f"| Within-task shuffle (single) | AUC | {wis.get('shuffled_auc', 'n/a')} "
        f"(drop {drop_s}) |"
    )
    boot = c["bootstrap_within_task"]
    if boot.get("ci_low") is not None:
        lines.append(
            f"| Bootstrap 95% CI (within-task AUC) | range | "
            f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] |"
        )
    global_dep = c.get("global_permutation_deprecated", {})
    lines.append(
        f"| Global shuffle (deprecated) | p-value | "
        f"{global_dep.get('p_value', 'n/a')} — confounded, do not interpret |"
    )
    lines.extend([
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "## How to read (v2)",
        "",
        "- **Within-task AUC (PRIMARY)**: predicts success vs failure among trajectories "
        "of the *same* task. Must be > 0.5 for latent forecasting signal.",
        "- **LOTO**: train on all tasks except one, test on held-out task. Kills task "
        "memorization across tasks.",
        "- **Within-task shuffle**: permutes labels within each task only. AUC → ~0.5 "
        "if clean; high AUC after shuffle = leakage.",
        "- **Text baseline (within-task)**: if activation ≈ text, signal is surface "
        "wording not internal state.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    trajectories = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))
    probe_payload = None
    if PROBE_PATH.exists():
        probe_payload = json.loads(PROBE_PATH.read_text(encoding="utf-8"))

    rows, features, labels, meta, n_layers, d_model = build_indexed_rows(trajectories)
    if n_layers is None:
        raise SystemExit("No cached activations — run Phase 1 first.")

    ref_layer = best_layer_early(probe_payload, features, labels, meta, n_layers)
    X = np.array(features[(ref_layer, EARLY_BIN)], dtype=np.float64)
    y = np.array(labels[EARLY_BIN], dtype=int)
    groups = np.array([m["instance_id"] for m in meta[EARLY_BIN]])
    traj_ids = [m["traj_id"] for m in meta[EARLY_BIN]]
    texts = [r["early_text"] for r in rows[EARLY_BIN]]
    texts_stripped = [r["early_text_stripped"] for r in rows[EARLY_BIN]]
    rng = np.random.default_rng(RANDOM_STATE)

    activation_wt = within_task_cv(X, y, groups)
    loto = loto_cv(X, y, groups)
    text_wt = text_baseline_within_task(texts, y, groups)
    text_stripped = text_baseline_within_task(texts_stripped, y, groups)
    within_perm = within_task_permutation_null(X, y, groups, N_PERM, rng)

    real_auc = activation_wt.get("micro_auc")
    y_inst = shuffle_within_instance(y, groups, rng)
    shuffled_wt = within_task_cv(X, y_inst, groups)
    shuffled_auc = shuffled_wt.get("micro_auc")
    auc_drop = (real_auc - shuffled_auc) if (real_auc is not None and shuffled_auc is not None) else None

    bootstrap = bootstrap_within_task_ci(X, y, groups, traj_ids, N_BOOT, rng)

    # Deprecated global check for comparison only
    from analysis.probe import probe_one as global_probe
    global_perm_real = global_probe(X, y)
    global_perm_aucs = []
    for _ in range(min(100, N_PERM)):
        y_perm = rng.permutation(y)
        cell = global_probe(X, y_perm)
        if cell.get("auc") is not None:
            global_perm_aucs.append(cell["auc"])
    global_p = None
    if global_perm_real.get("auc") is not None and global_perm_aucs:
        global_p = float((np.array(global_perm_aucs) >= global_perm_real["auc"]).mean())

    by_inst: dict[str, set[int]] = defaultdict(set)
    for r in rows[EARLY_BIN]:
        by_inst[r["instance_id"]].add(r["success"])
    n_mixed = sum(1 for s in by_inst.values() if len(s) > 1)

    checks = {
        "activation_within_task": activation_wt,
        "loto": loto,
        "text_baseline_within_task": text_wt,
        "text_baseline_stripped": text_stripped,
        "within_task_permutation": within_perm,
        "within_instance_shuffle": {
            "real_auc": real_auc,
            "shuffled_auc": shuffled_auc,
            "auc_drop": auc_drop,
            "n_instances": len(by_inst),
            "n_instances_with_both_labels": n_mixed,
        },
        "bootstrap_within_task": bootstrap,
        "global_permutation_deprecated": {
            "real_auc": global_perm_real.get("auc"),
            "p_value": global_p,
            "note": "deprecated confounded metric — do not interpret",
        },
    }

    report = {
        "version": "v2",
        "reference_layer": ref_layer,
        "reference_bin": BIN_NAMES[EARLY_BIN],
        "n_trajectories": len(trajectories),
        "n_instances": len(by_inst),
        "n_mixed_tasks": n_mixed,
        "d_model": d_model,
        "probe_C": PROBE_C,
        "checks": checks,
        "verdict": _verdict(checks),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")

    print(f"Skepticism checks v2 (early bin, layer {ref_layer})", flush=True)
    print(f"  within-task micro-AUC: {activation_wt.get('micro_auc')}", flush=True)
    print(f"  LOTO AUC: {loto.get('auc')}", flush=True)
    print(f"  text baseline within-task AUC: {text_wt.get('within_task_micro_auc')}", flush=True)
    print(f"  within-task permutation p-value: {within_perm.get('p_value')}", flush=True)
    print(f"  within-task shuffled AUC: {shuffled_auc} (drop {auc_drop})", flush=True)
    if bootstrap.get("ci_low") is not None:
        print(f"  bootstrap 95% CI: [{bootstrap['ci_low']:.3f}, {bootstrap['ci_high']:.3f}]",
              flush=True)
    print(f"\n{report['verdict']}", flush=True)
    print(f"\nSaved {OUT_PATH} and {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
