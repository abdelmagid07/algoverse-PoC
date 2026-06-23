"""Skepticism checks on probe results (run after analysis/probe.py).

Validates whether above-chance decodability survives controls that rule out
common confounds:

  1. Global label permutation null (luck on N trajectories)
  2. Within-instance label shuffle (task-ID / instance memorization)
  3. Instance holdout CV (generalize to unseen SWE-bench tasks)
  4. Early-text TF-IDF baseline vs activation probe (issue-text-only signal)
  5. Bootstrap CIs on trajectory resampling (early bin, reference layer)

Run:  python -m analysis.skepticism
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

from analysis.probe import (
    BIN_NAMES,
    PROBE_C,
    RANDOM_STATE,
    build_dataset,
    probe_one,
    relative_bin,
)
from interp.activation_cache import RESULTS_DIR, load_activations

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
PROBE_PATH = RESULTS_DIR / "probe_results.json"
OUT_PATH = RESULTS_DIR / "skepticism_report.json"
OUT_MD = RESULTS_DIR / "skepticism_report.md"

EARLY_BIN = 0
N_PERM = 300
N_BOOT = 400
MAX_TFIDF_FEATURES = 256


def _early_text(traj: dict) -> str:
    """Concatenate ai step previews that fall in the early relative bin."""
    texts = traj.get("step_texts") or []
    total = len(traj.get("step_positions") or texts)
    if total == 0:
        return ""
    parts = []
    for s, text in enumerate(texts[:total]):
        if relative_bin(s, total) == EARLY_BIN and text:
            parts.append(text)
    return " ".join(parts)


def build_indexed_rows(trajectories: list[dict]) -> tuple[dict, int, int]:
    """Per-bin rows aligned across trajectories for skepticism checks.

    Returns:
      rows: dict[bin] -> list of {traj_id, instance_id, success, early_text}
      features, labels, n_layers, d_model from build_dataset (same row order)
    """
    features, labels, n_layers, d_model = build_dataset(trajectories)
    rows: dict[int, list[dict]] = {b: [] for b in range(len(BIN_NAMES))}

    for traj in trajectories:
        try:
            load_activations(traj["id"])
        except FileNotFoundError:
            continue
        total = len(traj["step_positions"])
        meta = {
            "traj_id": traj["id"],
            "instance_id": str(traj.get("instance_id", traj["id"])),
            "success": int(bool(traj["success"])),
            "early_text": _early_text(traj),
        }
        seen_bins = set()
        for s in range(total):
            seen_bins.add(relative_bin(s, total))
        for b in sorted(seen_bins):
            rows[b].append(meta)

    return rows, features, labels, n_layers, d_model


def best_layer_early(probe_payload: dict | None, features, labels, n_layers: int) -> int:
    """Layer with highest early-bin CV accuracy in saved probe results, else argmax here."""
    if probe_payload:
        early = [r for r in probe_payload["results"]
                 if r["bin_idx"] == EARLY_BIN and r["acc_mean"] is not None]
        if early:
            return int(max(early, key=lambda r: r["acc_mean"])["layer"])
    best_l, best_acc = 0, -1.0
    for layer in range(n_layers):
        X = np.array(features[(layer, EARLY_BIN)], dtype=np.float64)
        y = np.array(labels[EARLY_BIN], dtype=int)
        cell = probe_one(X, y)
        if cell["acc_mean"] is not None and cell["acc_mean"] > best_acc:
            best_acc = cell["acc_mean"]
            best_l = layer
    return best_l


def shuffle_within_instance(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y2 = y.copy()
    for inst in np.unique(groups):
        idx = np.where(groups == inst)[0]
        if len(idx) > 1:
            y2[idx] = rng.permutation(y[idx])
    return y2


def permutation_null_auc(X: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator) -> dict:
    """Null distribution of AUC under global label shuffles."""
    real = probe_one(X, y)
    real_auc = real.get("auc")
    if real_auc is None:
        return {"real_auc": None, "null_mean": None, "p_value": None, "note": real.get("note", "")}

    null_aucs = []
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        cell = probe_one(X, y_perm)
        if cell.get("auc") is not None:
            null_aucs.append(cell["auc"])
    null_aucs = np.array(null_aucs)
    p = float((null_aucs >= real_auc).mean()) if len(null_aucs) else None
    return {
        "real_auc": float(real_auc),
        "null_mean": float(null_aucs.mean()) if len(null_aucs) else None,
        "null_std": float(null_aucs.std()) if len(null_aucs) else None,
        "p_value": p,
        "n_perm": len(null_aucs),
    }


def instance_holdout(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-instance-out CV: test on trajectories from unseen SWE-bench tasks."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return {"note": "need >= 2 instances for instance holdout", "acc_mean": None, "auc": None}
    try:
        return probe_one(X, y, cv=LeaveOneGroupOut(), groups=groups)
    except Exception as exc:  # pragma: no cover - e.g. single-class fold
        return {"note": f"instance holdout failed: {exc}", "acc_mean": None, "auc": None}


def text_baseline_early(texts: list[str], y: np.ndarray) -> dict:
    """TF-IDF on early-bin step text only (no activations)."""
    if len(texts) != len(y) or len(y) < 4:
        return {"note": "insufficient rows for text baseline"}
    vec = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, stop_words="english")
    try:
        X = vec.fit_transform(texts).toarray()
    except ValueError as exc:
        return {"note": f"tfidf failed: {exc}"}
    return probe_one(X, y)


def bootstrap_auc_ci(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> dict:
    """Bootstrap trajectories (rows) with replacement; distribution of OOF AUC."""
    n = len(y)
    if n < 4:
        return {"note": "too few rows for bootstrap"}
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Need both classes in resample
        if len(np.unique(y[idx])) < 2:
            continue
        cell = probe_one(X[idx], y[idx])
        if cell.get("auc") is not None:
            aucs.append(cell["auc"])
    if not aucs:
        return {"note": "no valid bootstrap samples"}
    aucs = np.array(aucs)
    lo = float(np.percentile(aucs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(aucs, (1 + ci) / 2 * 100))
    return {
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot_effective": int(len(aucs)),
    }


def _verdict(checks: dict) -> str:
    """Short human verdict from the control battery."""
    flags = []
    perm = checks.get("global_permutation", {})
    if perm.get("p_value") is not None and perm["p_value"] > 0.05:
        flags.append("global permutation not significant (p>0.05)")
    inst = checks.get("within_instance_shuffle", {})
    drop = inst.get("auc_drop")
    if drop is not None and drop < 0.05:
        flags.append(
            "within-instance label shuffle barely hurts AUC (task/instance cues may dominate)"
        )
    hold = checks.get("instance_holdout", {})
    if hold.get("auc") is not None and hold["auc"] < 0.55:
        flags.append("instance holdout AUC weak (<0.55)")
    text = checks.get("text_baseline_early", {})
    act = checks.get("activation_early", {})
    if text.get("auc") is not None and act.get("auc") is not None:
        if text["auc"] >= act["auc"] - 0.03:
            flags.append("early text alone nearly matches activations")
        elif act["auc"] - text["auc"] >= 0.08:
            flags.append("activations beat text baseline (internal signal beyond issue text)")

    if not flags:
        return (
            "Controls look favorable: signal survives instance holdout and/or beats "
            "text baseline; permutation p-value supports above-chance decoding."
        )
    return "Caveats: " + "; ".join(flags) + "."


def render_markdown(report: dict) -> str:
    c = report["checks"]
    lines = [
        "# Skepticism checks (auto-generated)",
        "",
        f"Reference: early bin, layer {report['reference_layer']}, "
        f"n={report['n_trajectories']} trajectories, "
        f"{report['n_instances']} distinct instances.",
        "",
        "## Results",
        "",
        "| Check | Metric | Value |",
        "|---|---|---|",
    ]
    act = c["activation_early"]
    lines.append(
        f"| Activation probe (early) | AUC | {act.get('auc', 'n/a')} "
        f"(acc {act.get('acc_mean', 'n/a')}) |"
    )
    txt = c["text_baseline_early"]
    lines.append(
        f"| Early text TF-IDF only | AUC | {txt.get('auc', 'n/a')} "
        f"(acc {txt.get('acc_mean', 'n/a')}) |"
    )
    perm = c["global_permutation"]
    lines.append(
        f"| Global label shuffle null | p-value | {perm.get('p_value', 'n/a')} "
        f"(null AUC mean {perm.get('null_mean', 'n/a')}) |"
    )
    wis = c["within_instance_shuffle"]
    drop_s = f"{wis['auc_drop']:+.3f}" if wis.get("auc_drop") is not None else "n/a"
    lines.append(
        f"| Within-instance label shuffle | AUC | {wis.get('shuffled_auc', 'n/a')} "
        f"(drop {drop_s} vs real) |"
    )
    hold = c["instance_holdout"]
    lines.append(
        f"| Instance holdout (LOIO) | AUC | {hold.get('auc', 'n/a')} "
        f"(acc {hold.get('acc_mean', 'n/a')}) |"
    )
    boot = c["bootstrap_early"]
    if boot.get("ci_low") is not None:
        lines.append(
            f"| Bootstrap 95% CI (early AUC) | range | "
            f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] |"
        )
    lines.extend([
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "## How to read",
        "",
        "- **Global permutation p-value**: fraction of shuffled-label runs with AUC "
        "≥ real. Low p → unlikely to be luck on N alone.",
        "- **Within-instance shuffle**: permutes success/fail labels among attempts "
        "at the *same* SWE-bench task. Large AUC drop → signal used instance-specific "
        "structure; small drop → more about per-attempt dynamics.",
        "- **Instance holdout**: train on 5 tasks, test on the 6th (leave-one-instance-out). "
        "Strong AUC → generalizes across repos/issues.",
        "- **Text baseline**: early `ai` step text only. If activation AUC ≫ text, "
        "internal state adds signal beyond obvious wording.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    trajectories = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))
    probe_payload = None
    if PROBE_PATH.exists():
        probe_payload = json.loads(PROBE_PATH.read_text(encoding="utf-8"))

    rows, features, labels, n_layers, d_model = build_indexed_rows(trajectories)
    if n_layers is None:
        raise SystemExit("No cached activations — run Phase 1 first.")

    ref_layer = best_layer_early(probe_payload, features, labels, n_layers)
    X = np.array(features[(ref_layer, EARLY_BIN)], dtype=np.float64)
    y = np.array(labels[EARLY_BIN], dtype=int)
    groups = np.array([r["instance_id"] for r in rows[EARLY_BIN]])
    texts = [r["early_text"] for r in rows[EARLY_BIN]]
    rng = np.random.default_rng(RANDOM_STATE)

    activation_early = probe_one(X, y)
    text_early = text_baseline_early(texts, y)
    global_perm = permutation_null_auc(X, y, N_PERM, rng)

    real_auc = activation_early.get("auc")
    y_inst = shuffle_within_instance(y, groups, rng)
    shuffled_cell = probe_one(X, y_inst)
    shuffled_auc = shuffled_cell.get("auc")
    auc_drop = (real_auc - shuffled_auc) if (real_auc is not None and shuffled_auc is not None) else None

    holdout = instance_holdout(X, y, groups)
    bootstrap = bootstrap_auc_ci(X, y, N_BOOT, rng)

    # Per-instance label diversity (how many instances have both classes?)
    by_inst: dict[str, set[int]] = defaultdict(set)
    for r in rows[EARLY_BIN]:
        by_inst[r["instance_id"]].add(r["success"])
    n_mixed = sum(1 for s in by_inst.values() if len(s) > 1)

    checks = {
        "activation_early": activation_early,
        "text_baseline_early": text_early,
        "global_permutation": global_perm,
        "within_instance_shuffle": {
            "real_auc": real_auc,
            "shuffled_auc": shuffled_auc,
            "auc_drop": auc_drop,
            "n_instances": len(by_inst),
            "n_instances_with_both_labels": n_mixed,
            "note": (
                "shuffle only permutes within instances that have >= 2 attempts; "
                f"{n_mixed}/{len(by_inst)} instances have both success and fail."
            ),
        },
        "instance_holdout": holdout,
        "bootstrap_early": bootstrap,
    }

    report = {
        "reference_layer": ref_layer,
        "reference_bin": BIN_NAMES[EARLY_BIN],
        "n_trajectories": len(trajectories),
        "n_instances": len(by_inst),
        "d_model": d_model,
        "probe_C": PROBE_C,
        "checks": checks,
        "verdict": _verdict(checks),
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")

    print(f"Skepticism checks (early bin, layer {ref_layer})", flush=True)
    print(f"  activation AUC: {activation_early.get('auc')}", flush=True)
    print(f"  text baseline AUC: {text_early.get('auc')}", flush=True)
    print(f"  permutation p-value: {global_perm.get('p_value')}", flush=True)
    print(f"  within-instance shuffled AUC: {shuffled_auc} (drop {auc_drop})", flush=True)
    print(f"  instance holdout AUC: {holdout.get('auc')}", flush=True)
    if bootstrap.get("ci_low") is not None:
        print(f"  bootstrap 95% CI: [{bootstrap['ci_low']:.3f}, {bootstrap['ci_high']:.3f}]",
              flush=True)
    print(f"\n{report['verdict']}", flush=True)
    print(f"\nSaved {OUT_PATH} and {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
