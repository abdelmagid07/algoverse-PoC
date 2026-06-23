"""Phase 2: linear probes for latent failure forecasting (v2).

Core question: can a simple logistic-regression probe on the residual stream
predict success vs failure **within the same task** (different trajectories)?

v2 primary metrics:
  * within_task_auc — stratified CV inside each task, pooled OOF
  * loto_auc — leave-one-task-out generalization
  * global_auc — deprecated confounded baseline (kept for comparison)

Run:  python -m analysis.probe
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.dataset_validate import validate_dataset
from interp.activation_cache import RESULTS_DIR, load_activations

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
OUT_PATH = RESULTS_DIR / "probe_results.json"

N_BINS = 3
BIN_NAMES = ["early", "mid", "late"]
MAX_FOLDS = 5
PROBE_C = 0.1
RANDOM_STATE = 0
MAX_PAIRS_PER_TASK = int(os.environ.get("VERITAS_MAX_PAIRS_PER_TASK", "20"))


def relative_bin(step_idx: int, total_steps: int) -> int:
    """Map a 0-indexed step to a relative-position bin in [0, N_BINS)."""
    if total_steps <= 1:
        return 0
    return min(N_BINS - 1, int(N_BINS * step_idx / total_steps))


def build_dataset(trajectories: list[dict]):
    """Build per-(layer, bin) feature matrices and aligned metadata.

    Returns:
      features: dict[(layer, bin)] -> list[np.ndarray]
      labels:   dict[bin] -> list[int]
      meta:     dict[bin] -> list[dict] with traj_id, instance_id
      n_layers, d_model
    """
    features: dict[tuple[int, int], list[np.ndarray]] = {}
    labels: dict[int, list[int]] = {b: [] for b in range(N_BINS)}
    meta: dict[int, list[dict]] = {b: [] for b in range(N_BINS)}
    n_layers = None
    d_model = None

    for traj in trajectories:
        try:
            acts = load_activations(traj["id"])
        except FileNotFoundError:
            print(f"  warning: no cached activations for {traj['id'][:8]}, skipping", flush=True)
            continue

        layer_names = sorted(acts.keys(), key=lambda n: int(n.split(".")[1]))
        if n_layers is None:
            n_layers = len(layer_names)
            d_model = int(acts[layer_names[0]].shape[-1])
            for layer in range(n_layers):
                for b in range(N_BINS):
                    features[(layer, b)] = []

        step_positions = traj["step_positions"]
        total = len(step_positions)
        label = int(bool(traj["success"]))
        inst_id = str(traj.get("instance_id", traj["id"]))

        bin_to_steps: dict[int, list[int]] = {}
        for s in range(total):
            bin_to_steps.setdefault(relative_bin(s, total), []).append(s)

        for b, step_idxs in bin_to_steps.items():
            labels[b].append(label)
            meta[b].append({"traj_id": traj["id"], "instance_id": inst_id})
            positions = [step_positions[s] for s in step_idxs]
            for layer, name in enumerate(layer_names):
                act = acts[name].float().numpy()
                pooled = act[positions, :].mean(axis=0)
                features[(layer, b)].append(pooled)

    return features, labels, meta, n_layers, d_model


def _folds_for(y: np.ndarray) -> int:
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    min_class = counts.min()
    return int(min(MAX_FOLDS, min_class))


def _make_clf():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=PROBE_C, max_iter=2000, penalty="l2"),
    )


def probe_one(X: np.ndarray, y: np.ndarray, cv=None, groups: np.ndarray | None = None) -> dict:
    """Cross-validated probe for a single (layer, bin) cell."""
    n = len(y)
    n_pos = int(y.sum())
    n_neg = int(n - n_pos)
    chance = max(n_pos, n_neg) / n if n else float("nan")

    out = {
        "n": n, "n_pos": n_pos, "n_neg": n_neg, "chance": chance,
        "acc_mean": None, "acc_std": None, "auc": None, "folds": 0,
        "note": "",
    }

    if n_pos < 2 or n_neg < 2:
        out["note"] = "insufficient class balance for stratified CV"
        return out

    k = _folds_for(y)
    if cv is None:
        if k < 2:
            out["note"] = "too few per-class examples for CV"
            return out
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        out["folds"] = int(k)
    else:
        if groups is not None:
            out["folds"] = int(cv.get_n_splits(X, y, groups=groups))
        else:
            out["folds"] = int(cv.get_n_splits(X, y))
        if out["folds"] < 2:
            out["note"] = "CV splitter has fewer than 2 folds"
            return out

    clf = _make_clf()
    accs = cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="accuracy")
    out["acc_mean"] = float(accs.mean())
    out["acc_std"] = float(accs.std())
    if groups is not None and out["folds"] == 0:
        out["folds"] = len(accs)

    try:
        proba = cross_val_predict(
            clf, X, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]
        out["auc"] = float(roc_auc_score(y, proba))
    except Exception as exc:
        out["note"] = (out["note"] + f"; auc failed: {exc}").strip("; ")

    if not np.isfinite(accs).all():
        out["note"] = (out["note"] + "; NaN accuracy").strip("; ")

    return out


def within_task_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Primary metric: stratified CV within each task, pooled OOF predictions."""
    unique_groups = np.unique(groups)
    oof_proba = np.full(len(y), np.nan)
    oof_mask = np.zeros(len(y), dtype=bool)
    per_task_auc: list[float] = []

    for inst in unique_groups:
        idx = np.where(groups == inst)[0]
        y_task = y[idx]
        n_pos = int(y_task.sum())
        n_neg = len(y_task) - n_pos
        if n_pos < 1 or n_neg < 1:
            continue
        if len(idx) < 4 or n_pos < 2 or n_neg < 2:
            # Too few for CV — use leave-one-out style within task
            for i, row_idx in enumerate(idx):
                train_idx = np.delete(idx, i)
                if len(np.unique(y[train_idx])) < 2:
                    continue
                clf = _make_clf()
                clf.fit(X[train_idx], y[train_idx])
                oof_proba[row_idx] = clf.predict_proba(X[row_idx : row_idx + 1])[0, 1]
                oof_mask[row_idx] = True
            task_mask = oof_mask[idx]
            if task_mask.sum() >= 2 and len(np.unique(y_task[task_mask])) == 2:
                try:
                    per_task_auc.append(
                        float(roc_auc_score(y_task[task_mask], oof_proba[idx][task_mask]))
                    )
                except ValueError:
                    pass
            continue

        k = _folds_for(y_task)
        if k < 2:
            continue
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
        clf = _make_clf()
        try:
            proba = cross_val_predict(
                clf, X[idx], y_task, cv=cv, method="predict_proba"
            )[:, 1]
            oof_proba[idx] = proba
            oof_mask[idx] = True
            per_task_auc.append(float(roc_auc_score(y_task, proba)))
        except Exception:
            continue

    out = {
        "micro_auc": None,
        "macro_auc": None,
        "macro_auc_median": None,
        "macro_auc_iqr": None,
        "n_tasks_evaluated": len(per_task_auc),
        "per_task_aucs": per_task_auc,
        "note": "",
    }

    if oof_mask.sum() < 4 or len(np.unique(y[oof_mask])) < 2:
        out["note"] = "insufficient within-task OOF predictions"
        return out

    out["micro_auc"] = float(roc_auc_score(y[oof_mask], oof_proba[oof_mask]))
    if per_task_auc:
        arr = np.array(per_task_auc)
        out["macro_auc"] = float(arr.mean())
        out["macro_auc_median"] = float(np.median(arr))
        q75, q25 = np.percentile(arr, [75, 25])
        out["macro_auc_iqr"] = float(q75 - q25)

    return out


def loto_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """Leave-one-task-out CV."""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return {"auc": None, "acc_mean": None, "note": "need >= 2 tasks for LOTO"}
    return probe_one(X, y, cv=LeaveOneGroupOut(), groups=groups)


def paired_difference_probe(
    trajectories: list[dict],
    layer: int,
    bin_idx: int,
    layer_names: list[str] | None = None,
) -> dict:
    """Train on Δh = h(success) - h(failure) pairs within each task."""
    from collections import defaultdict

    by_task: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)

    for traj in trajectories:
        try:
            acts = load_activations(traj["id"])
        except FileNotFoundError:
            continue
        if layer_names is None:
            layer_names = sorted(acts.keys(), key=lambda n: int(n.split(".")[1]))
        step_positions = traj["step_positions"]
        total = len(step_positions)
        bin_to_steps: dict[int, list[int]] = {}
        for s in range(total):
            bin_to_steps.setdefault(relative_bin(s, total), []).append(s)
        if bin_idx not in bin_to_steps:
            continue
        positions = [step_positions[s] for s in bin_to_steps[bin_idx]]
        name = layer_names[layer]
        act = acts[name].float().numpy()
        pooled = act[positions, :].mean(axis=0)
        label = int(bool(traj["success"]))
        inst = str(traj.get("instance_id", traj["id"]))
        by_task[inst].append((label, pooled))

    delta_X: list[np.ndarray] = []
    delta_y: list[int] = []
    task_ids: list[str] = []

    for inst, items in by_task.items():
        succ = [h for lbl, h in items if lbl == 1]
        fail = [h for lbl, h in items if lbl == 0]
        if not succ or not fail:
            continue
        pairs = [(s, f) for s in succ for f in fail]
        if len(pairs) > MAX_PAIRS_PER_TASK:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(pairs), MAX_PAIRS_PER_TASK, replace=False)
            pairs = [pairs[i] for i in idx]
        for s_h, f_h in pairs:
            delta_X.append(s_h - f_h)
            delta_y.append(1)
            task_ids.append(inst)
            delta_X.append(f_h - s_h)
            delta_y.append(0)
            task_ids.append(inst)

    if len(delta_y) < 4 or len(set(delta_y)) < 2:
        return {"auc": None, "n_pairs": len(delta_y), "note": "insufficient pairs"}

    X = np.array(delta_X, dtype=np.float64)
    y = np.array(delta_y, dtype=int)
    groups = np.array(task_ids)

    # LOTO on task groups for paired differences
    loto = loto_cv(X, y, groups)
    return {
        "auc": loto.get("auc"),
        "acc_mean": loto.get("acc_mean"),
        "n_pairs": len(delta_y),
        "n_tasks": len(by_task),
        "note": loto.get("note", ""),
    }


def main() -> None:
    smoke = os.environ.get("VERITAS_SMOKE_N") is not None
    raw_trajectories = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))
    trajectories = validate_dataset(raw_trajectories, smoke=smoke)

    n_traj = len(trajectories)
    n_success = sum(bool(t["success"]) for t in trajectories)
    print(f"Trajectories (after validation): {n_traj} (success={n_success}, fail={n_traj - n_success})",
          flush=True)

    features, labels, meta, n_layers, d_model = build_dataset(trajectories)
    if n_layers is None:
        raise SystemExit("No cached activations found — run Phase 1 first.")
    print(f"Probing {n_layers} layers x {N_BINS} bins (d_model={d_model}, C={PROBE_C})", flush=True)

    overall_chance = max(n_success, n_traj - n_success) / n_traj if n_traj else float("nan")

    results = []
    for layer in range(n_layers):
        for b in range(N_BINS):
            X = np.array(features[(layer, b)], dtype=np.float64)
            y = np.array(labels[b], dtype=int)
            groups = np.array([m["instance_id"] for m in meta[b]])

            global_cell = probe_one(X, y)
            within_cell = within_task_cv(X, y, groups)
            loto_cell = loto_cv(X, y, groups)

            cell = {
                "layer": layer,
                "bin": BIN_NAMES[b],
                "bin_idx": b,
                "n": global_cell["n"],
                "n_pos": global_cell["n_pos"],
                "n_neg": global_cell["n_neg"],
                "chance": global_cell["chance"],
                "global_auc": global_cell.get("auc"),
                "global_acc_mean": global_cell.get("acc_mean"),
                "global_acc_std": global_cell.get("acc_std"),
                "global_folds": global_cell.get("folds", 0),
                "global_note": global_cell.get("note", ""),
                "metric_tier_global": "deprecated_confounded",
                "within_task_micro_auc": within_cell.get("micro_auc"),
                "within_task_macro_auc": within_cell.get("macro_auc"),
                "within_task_macro_median": within_cell.get("macro_auc_median"),
                "within_task_macro_iqr": within_cell.get("macro_auc_iqr"),
                "within_task_n_tasks": within_cell.get("n_tasks_evaluated"),
                "within_task_note": within_cell.get("note", ""),
                "loto_auc": loto_cell.get("auc"),
                "loto_acc_mean": loto_cell.get("acc_mean"),
                "loto_note": loto_cell.get("note", ""),
                # Legacy aliases for visualize/summarize compatibility
                "acc_mean": global_cell.get("acc_mean"),
                "acc_std": global_cell.get("acc_std"),
                "auc": within_cell.get("micro_auc"),
            }
            results.append(cell)

    # Paired-difference probes at each layer/bin
    layer_names = None
    paired_results = []
    for layer in range(n_layers):
        for b in range(N_BINS):
            pd = paired_difference_probe(trajectories, layer, b, layer_names)
            pd["layer"] = layer
            pd["bin"] = BIN_NAMES[b]
            pd["bin_idx"] = b
            paired_results.append(pd)

    payload = {
        "version": "v2",
        "n_trajectories": n_traj,
        "n_success": n_success,
        "n_fail": n_traj - n_success,
        "overall_chance": overall_chance,
        "n_layers": n_layers,
        "d_model": d_model,
        "bins": BIN_NAMES,
        "max_folds": MAX_FOLDS,
        "probe_C": PROBE_C,
        "primary_metric": "within_task_micro_auc",
        "results": results,
        "paired_difference": paired_results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nChance baseline (majority class): {overall_chance:.3f}", flush=True)
    for b, name in enumerate(BIN_NAMES):
        cells = [r for r in results if r["bin_idx"] == b and r["within_task_micro_auc"] is not None]
        if cells:
            best = max(cells, key=lambda r: r["within_task_micro_auc"] or 0)
            print(
                f"  {name:5}: n={best['n']:>3}  best layer {best['layer']:>2}  "
                f"within_task_auc={best['within_task_micro_auc']:.3f}  "
                f"loto_auc={best['loto_auc'] if best['loto_auc'] is None else round(best['loto_auc'], 3)}  "
                f"global_auc={best['global_auc'] if best['global_auc'] is None else round(best['global_auc'], 3)} (deprecated)",
                flush=True,
            )
        else:
            print(f"  {name:5}: no probeable cells", flush=True)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
