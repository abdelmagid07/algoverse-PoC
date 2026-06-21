"""Phase 2: linear probes for latent failure forecasting.

Core question: can a simple logistic-regression probe on the residual stream
predict a trajectory's eventual success/failure, and does that decodability
strengthen as the agent nears its own conclusion?

Design (see plan):
  * Step axis is RELATIVE position. Each trajectory's steps are split into three
    bins (first / middle / final third by step_idx / total_steps), so a 3-step
    and a 6-step trajectory both contribute to every bin. This measures "signal
    grows as the agent approaches its own answer," not "long trajectories differ
    from short ones," and keeps the per-bin sample size roughly constant.
  * ONE row per (trajectory, bin): mean-pool the activations of the steps that
    fall in a bin. So each probe sees exactly one example per trajectory, and CV
    folds split cleanly by trajectory (no step-level leakage).
  * EVERY layer is probed (no cherry-picking a layer post hoc). Output is a
    layer x bin grid.
  * High-dim/low-N hygiene: the residual stream is ~2048-dim but N ~ 18, so a
    raw probe would trivially overfit. We use StandardScaler + L2-regularized
    LogisticRegression in a Pipeline, with the scaler fit on the train fold only.
    Absolute accuracy is therefore regularization-sensitive; the summary flags it.
  * Honest metrics: StratifiedKFold (k auto-reduced if a class is small),
    per-fold accuracy mean +/- std, AUC from pooled out-of-fold probabilities,
    and a majority-class chance baseline (not assumed 0.5).

Run:  python -m analysis.probe
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from interp.activation_cache import RESULTS_DIR, load_activations

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
OUT_PATH = RESULTS_DIR / "probe_results.json"

N_BINS = 3
BIN_NAMES = ["early", "mid", "late"]
MAX_FOLDS = 5
PROBE_C = 0.1   # strong L2 regularization for the p >> n regime
RANDOM_STATE = 0


def relative_bin(step_idx: int, total_steps: int) -> int:
    """Map a 0-indexed step to a relative-position bin in [0, N_BINS)."""
    if total_steps <= 1:
        return 0
    return min(N_BINS - 1, int(N_BINS * step_idx / total_steps))


def build_dataset(trajectories: list[dict]):
    """Build per-(layer, bin) feature matrices and per-bin labels.

    Returns:
      features: dict[(layer, bin)] -> list[np.ndarray]   (rows aligned with labels[bin])
      labels:   dict[bin] -> list[int]
      n_layers, d_model
    """
    features: dict[tuple[int, int], list[np.ndarray]] = {}
    labels: dict[int, list[int]] = {b: [] for b in range(N_BINS)}
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

        # Group step indices by relative bin.
        bin_to_steps: dict[int, list[int]] = {}
        for s in range(total):
            bin_to_steps.setdefault(relative_bin(s, total), []).append(s)

        for b, step_idxs in bin_to_steps.items():
            labels[b].append(label)
            positions = [step_positions[s] for s in step_idxs]
            for layer, name in enumerate(layer_names):
                act = acts[name].float().numpy()  # [seq, d_model]
                pooled = act[positions, :].mean(axis=0)  # mean-pool steps in bin
                features[(layer, b)].append(pooled)

    return features, labels, n_layers, d_model


def _folds_for(y: np.ndarray) -> int:
    """Largest valid fold count: <= MAX_FOLDS and <= smallest class count."""
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    min_class = counts.min()
    return int(min(MAX_FOLDS, min_class))


def probe_one(X: np.ndarray, y: np.ndarray) -> dict:
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
    if k < 2:
        out["note"] = "too few per-class examples for CV"
        return out

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=PROBE_C, max_iter=2000, penalty="l2"),
    )
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)

    accs = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    out["acc_mean"] = float(accs.mean())
    out["acc_std"] = float(accs.std())
    out["folds"] = int(k)

    # Pooled out-of-fold AUC is more stable than averaging tiny per-fold AUCs.
    try:
        proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        out["auc"] = float(roc_auc_score(y, proba))
    except Exception as exc:  # pragma: no cover - defensive
        out["note"] = (out["note"] + f"; auc failed: {exc}").strip("; ")

    if not np.isfinite(accs).all():
        out["note"] = (out["note"] + "; NaN accuracy").strip("; ")

    return out


def main() -> None:
    trajectories = json.loads(TRAJ_PATH.read_text(encoding="utf-8"))
    n_traj = len(trajectories)
    n_success = sum(bool(t["success"]) for t in trajectories)
    print(f"Trajectories: {n_traj} (success={n_success}, fail={n_traj - n_success})", flush=True)

    features, labels, n_layers, d_model = build_dataset(trajectories)
    if n_layers is None:
        raise SystemExit("No cached activations found — run Phase 1 first.")
    print(f"Probing {n_layers} layers x {N_BINS} bins (d_model={d_model}, C={PROBE_C})", flush=True)

    overall_chance = max(n_success, n_traj - n_success) / n_traj if n_traj else float("nan")

    results = []
    for layer in range(n_layers):
        for b in range(N_BINS):
            X = np.array(features[(layer, b)], dtype=np.float64)
            y = np.array(labels[b], dtype=int)
            cell = probe_one(X, y)
            cell["layer"] = layer
            cell["bin"] = BIN_NAMES[b]
            cell["bin_idx"] = b
            results.append(cell)

    payload = {
        "n_trajectories": n_traj,
        "n_success": n_success,
        "n_fail": n_traj - n_success,
        "overall_chance": overall_chance,
        "n_layers": n_layers,
        "d_model": d_model,
        "bins": BIN_NAMES,
        "max_folds": MAX_FOLDS,
        "probe_C": PROBE_C,
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Console sanity summary: best accuracy per bin across layers.
    print(f"\nChance baseline (majority class): {overall_chance:.3f}", flush=True)
    for b, name in enumerate(BIN_NAMES):
        cells = [r for r in results if r["bin_idx"] == b and r["acc_mean"] is not None]
        if cells:
            best = max(cells, key=lambda r: r["acc_mean"])
            print(f"  {name:5}: n={best['n']:>2}  best layer {best['layer']:>2} "
                  f"acc={best['acc_mean']:.3f}+/-{best['acc_std']:.3f} "
                  f"auc={best['auc'] if best['auc'] is None else round(best['auc'], 3)}",
                  flush=True)
        else:
            print(f"  {name:5}: no probeable cells (class balance too small)", flush=True)
    print(f"\nSaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
