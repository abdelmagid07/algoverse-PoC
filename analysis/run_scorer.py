"""Auto-scorer for v2 pipeline runs: INVALID / INCONCLUSIVE / MEANINGFUL.

Reads results/trajectories.json, probe_results.json, and skepticism_report.json
and applies the v2 stop-conditions checklist.

Run:  python -m analysis.run_scorer
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from analysis.dataset_validate import group_by_instance_id
from agent.sandbox_env import initial_user_message
from interp.activation_cache import RESULTS_DIR

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
PROBE_PATH = RESULTS_DIR / "probe_results.json"
SKEPTICISM_PATH = RESULTS_DIR / "skepticism_report.json"
OUT_JSON = RESULTS_DIR / "run_score.json"
OUT_MD = RESULTS_DIR / "run_score.md"
HISTORY_PATH = RESULTS_DIR / "run_score_history.jsonl"

# Thresholds from v2 stop-conditions spec
SHUFFLE_INVALID_AUC = 0.60
SHUFFLE_MEANINGFUL_LOW = 0.45
SHUFFLE_MEANINGFUL_HIGH = 0.55
AUC_WEAK_HIGH = 0.55
AUC_SUSPICIOUS = 0.70
AUC_MEANINGFUL = 0.55
LOTO_MEANINGFUL = 0.55
TEXT_MARGIN = 0.03
TASK_DOMINANCE_FRAC = 0.10
PER_TASK_AUC_VARIANCE = 0.20
MIN_TASKS_INCONCLUSIVE = 20
MIN_TASKS_MEANINGFUL = 20
MIN_TRAJ_INCONCLUSIVE = 100
MIN_K_INCONCLUSIVE = 3
MIN_K_MEANINGFUL = 4
SEVERE_IMBALANCE_RATIO = 0.75


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _check_task_id_leakage(trajectories: list[dict]) -> list[str]:
    reasons = []
    src = inspect.getsource(initial_user_message)
    if re.search(r"Task\s*ID\s*:", src, re.I):
        reasons.append("Task ID still present in initial_user_message() source")

    task_id_pat = re.compile(r"Task\s*ID\s*:", re.I)
    for t in trajectories:
        for field in ("step_texts",):
            for text in t.get(field) or []:
                if task_id_pat.search(text):
                    reasons.append(
                        f"Task ID pattern found in trajectory {t.get('id', '?')[:24]} step text"
                    )
                    return reasons
        inst = str(t.get("instance_id", ""))
        if inst:
            for text in t.get("step_texts") or []:
                if inst in text and inst.startswith("fix_"):
                    reasons.append(
                        f"instance_id '{inst}' appears in step text of {t.get('id', '?')[:24]}"
                    )
                    return reasons
    return reasons


def _check_grouping_and_duplicates(trajectories: list[dict]) -> list[str]:
    reasons = []
    ids_seen: dict[str, str] = {}
    pair_seen: dict[tuple[str, int], str] = {}

    for t in trajectories:
        tid = t.get("id")
        inst = t.get("instance_id")
        seed = t.get("seed", 0)

        if not inst:
            reasons.append(f"trajectory {tid} missing instance_id")
            continue

        if tid in ids_seen:
            if ids_seen[tid] != inst:
                reasons.append(f"trajectory id {tid} mapped to multiple instance_ids")
        else:
            ids_seen[tid] = inst

        key = (str(inst), int(seed))
        if key in pair_seen and pair_seen[key] != tid:
            reasons.append(
                f"duplicate (instance_id, seed) pair {key}: {pair_seen[key]} vs {tid}"
            )
        pair_seen[key] = tid

    dup_ids = [tid for tid, count in Counter(t.get("id") for t in trajectories).items() if count > 1]
    for tid in dup_ids:
        reasons.append(f"duplicate trajectory id in dataset: {tid}")

    return reasons


def _check_single_class_tasks(trajectories: list[dict]) -> list[str]:
    reasons = []
    by_task = group_by_instance_id(trajectories)
    for inst, ts in by_task.items():
        labels = {bool(t["success"]) for t in ts}
        if len(labels) < 2:
            label = "success" if True in labels else "failure"
            reasons.append(f"task {inst} has only {label} trajectories (single-class)")
    return reasons


def _check_labels_present(trajectories: list[dict]) -> list[str]:
    reasons = []
    for t in trajectories:
        if "success" not in t:
            reasons.append(f"trajectory {t.get('id', '?')} missing success label")
        elif not isinstance(t["success"], (bool, int)):
            reasons.append(f"trajectory {t.get('id', '?')} has non-boolean success")
    return reasons


def _check_seed_diversity(trajectories: list[dict]) -> list[str]:
    """Flag possible determinism bug if all seeds produce identical outputs."""
    reasons = []
    by_task = group_by_instance_id(trajectories)
    for inst, ts in by_task.items():
        if len(ts) < 2:
            continue
        fingerprints = []
        for t in ts:
            texts = t.get("step_texts") or []
            fp = hashlib.md5("".join(texts).encode()).hexdigest()
            fingerprints.append((t.get("seed", -1), fp, t.get("success")))
        unique_fps = {fp for _, fp, _ in fingerprints}
        unique_success = {s for _, _, s in fingerprints}
        if len(unique_fps) == 1 and len(ts) >= 2:
            reasons.append(
                f"task {inst}: all {len(ts)} seeds produce identical step text (determinism bug?)"
            )
        if len(unique_success) == 1 and len(ts) >= 3:
            # Could be legitimate but worth noting only if texts also identical
            if len(unique_fps) == 1:
                pass  # already flagged
    return reasons


def _check_do_sample_enabled() -> list[str]:
    reasons = []
    try:
        from agent import sandbox_runner
        src = inspect.getsource(sandbox_runner._generate_ai_turn)
        if "do_sample=True" not in src.replace(" ", "") and "do_sample = True" not in src:
            if "do_sample=False" in src or "do_sample = False" in src:
                reasons.append("sandbox_runner uses do_sample=False (deterministic generation)")
    except Exception:
        pass
    return reasons


def _check_evaluation(skepticism: dict | None, probe: dict | None) -> tuple[list[str], dict]:
    """Return (invalid_reasons, metrics)."""
    invalid = []
    metrics: dict = {}

    if skepticism is None:
        invalid.append("skepticism_report.json missing — within-task checks not run")
        return invalid, metrics

    checks = skepticism.get("checks", {})
    shuffle = checks.get("within_instance_shuffle", {})
    perm = checks.get("within_task_permutation", {})
    loto_sk = checks.get("loto", {})

    shuffled_auc = shuffle.get("shuffled_auc")
    auc_drop = shuffle.get("auc_drop")
    real_auc = shuffle.get("real_auc")

    metrics["shuffle_real_auc"] = real_auc
    metrics["shuffle_shuffled_auc"] = shuffled_auc
    metrics["shuffle_auc_drop"] = auc_drop

    if shuffled_auc is None:
        invalid.append("within-task shuffle test not run or failed")
    elif shuffled_auc > SHUFFLE_INVALID_AUC:
        invalid.append(
            f"shuffle test did NOT reduce performance (shuffled AUC={shuffled_auc:.3f} > {SHUFFLE_INVALID_AUC})"
        )
    elif auc_drop is not None and auc_drop < 0.10:
        invalid.append(
            f"shuffle test barely reduced AUC (drop={auc_drop:.3f}) — possible task-identity leakage"
        )

    if not perm or perm.get("p_value") is None:
        if perm.get("note"):
            invalid.append(f"within-task permutation not run: {perm.get('note')}")
        else:
            invalid.append("within-task permutation test not run")

    metrics["perm_p_value"] = perm.get("p_value")
    metrics["perm_null_mean"] = perm.get("null_mean")

    if probe is None:
        invalid.append("probe_results.json missing — LOTO cannot be verified")
        return invalid, metrics

    loto_values = [
        r["loto_auc"] for r in probe.get("results", [])
        if r.get("loto_auc") is not None
    ]
    if not loto_values:
        invalid.append("LOTO AUC absent from all probe cells — LOTO not implemented or failed")
    metrics["loto_auc_best"] = max(loto_values) if loto_values else None
    metrics["loto_auc_mean"] = float(sum(loto_values) / len(loto_values)) if loto_values else None

    if probe.get("version") != "v2" and probe.get("primary_metric") != "within_task_micro_auc":
        invalid.append("probe_results.json is not v2 format (missing within-task primary metrics)")

    return invalid, metrics


def _dataset_stats(trajectories: list[dict]) -> dict:
    by_task = group_by_instance_id(trajectories)
    counts = [len(ts) for ts in by_task.values()]
    k_min = min(counts) if counts else 0
    k_max = max(counts) if counts else 0
    n_tasks = len(by_task)
    n_traj = len(trajectories)

    imbalance_tasks = []
    for inst, ts in by_task.items():
        n_succ = sum(1 for t in ts if t["success"])
        n_fail = len(ts) - n_succ
        majority = max(n_succ, n_fail) / len(ts)
        if majority >= SEVERE_IMBALANCE_RATIO:
            imbalance_tasks.append(inst)

    dominant_task = None
    dominant_frac = 0.0
    if n_traj:
        for inst, ts in by_task.items():
            frac = len(ts) / n_traj
            if frac > dominant_frac:
                dominant_frac = frac
                dominant_task = inst

    return {
        "n_tasks": n_tasks,
        "n_trajectories": n_traj,
        "k_min": k_min,
        "k_max": k_max,
        "k_mean": sum(counts) / len(counts) if counts else 0,
        "imbalance_tasks": imbalance_tasks,
        "dominant_task": dominant_task,
        "dominant_frac": dominant_frac,
        "mixed_tasks": sum(
            1 for ts in by_task.values()
            if any(t["success"] for t in ts) and any(not t["success"] for t in ts)
        ),
    }


def _probe_metrics(probe: dict | None) -> dict:
    if not probe:
        return {}
    within_macro = [
        r["within_task_macro_auc"] for r in probe.get("results", [])
        if r.get("within_task_macro_auc") is not None
    ]
    within_micro = [
        r["within_task_micro_auc"] for r in probe.get("results", [])
        if r.get("within_task_micro_auc") is not None
    ]
    loto = [r["loto_auc"] for r in probe.get("results", []) if r.get("loto_auc") is not None]

    paired = probe.get("paired_difference") or []
    paired_auc = [p["auc"] for p in paired if p.get("auc") is not None]

    return {
        "within_macro_best": max(within_macro) if within_macro else None,
        "within_macro_mean": float(sum(within_macro) / len(within_macro)) if within_macro else None,
        "within_micro_best": max(within_micro) if within_micro else None,
        "loto_best": max(loto) if loto else None,
        "paired_auc_best": max(paired_auc) if paired_auc else None,
        "n_layers_with_auc_above_55": sum(
            1 for r in probe.get("results", [])
            if r.get("bin_idx") == 0
            and r.get("within_task_macro_auc") is not None
            and r["within_task_macro_auc"] >= AUC_MEANINGFUL
        ),
    }


def _skepticism_metrics(skepticism: dict | None) -> dict:
    if not skepticism:
        return {}
    checks = skepticism.get("checks", {})
    act = checks.get("activation_within_task", {})
    text = checks.get("text_baseline_within_task", {})
    text_s = checks.get("text_baseline_stripped", {})
    per_task = act.get("per_task_aucs") or []

    text_auc = text.get("within_task_micro_auc") or text_s.get("within_task_micro_auc")

    variance = None
    if len(per_task) >= 2:
        import numpy as np
        variance = float(np.std(per_task))

    return {
        "within_task_micro_auc": act.get("micro_auc"),
        "within_task_macro_auc": act.get("macro_auc"),
        "per_task_aucs": per_task,
        "per_task_auc_std": variance,
        "text_baseline_auc": text_auc,
        "loto_auc": checks.get("loto", {}).get("auc"),
    }


def _inconclusive_reasons(
    ds: dict,
    probe_m: dict,
    sk_m: dict,
    metrics: dict,
) -> list[str]:
    reasons = []

    if ds["k_min"] < MIN_K_INCONCLUSIVE:
        reasons.append(f"K < {MIN_K_INCONCLUSIVE} per task (min K={ds['k_min']})")
    if ds["n_tasks"] < MIN_TASKS_INCONCLUSIVE:
        reasons.append(f"< {MIN_TASKS_INCONCLUSIVE} tasks ({ds['n_tasks']})")
    if ds["n_trajectories"] < MIN_TRAJ_INCONCLUSIVE:
        reasons.append(f"< {MIN_TRAJ_INCONCLUSIVE} trajectories ({ds['n_trajectories']})")
    if ds["imbalance_tasks"]:
        reasons.append(
            f"severe within-task class imbalance in {len(ds['imbalance_tasks'])} tasks "
            f"(≥{SEVERE_IMBALANCE_RATIO:.0%} one class)"
        )

    wt_macro = sk_m.get("within_task_macro_auc") or probe_m.get("within_macro_best")
    wt_micro = sk_m.get("within_task_micro_auc") or probe_m.get("within_micro_best")
    loto = sk_m.get("loto_auc") or probe_m.get("loto_best")

    if wt_macro is not None and AUC_WEAK_HIGH >= wt_macro >= 0.5:
        reasons.append(f"within-task macro AUC weak ({wt_macro:.3f} ≈ chance)")
    if wt_micro is not None and wt_micro < AUC_MEANINGFUL:
        reasons.append(f"within-task micro AUC below meaningful threshold ({wt_micro:.3f} < {AUC_MEANINGFUL})")
    if loto is not None and loto < AUC_MEANINGFUL:
        reasons.append(f"LOTO AUC ≈ chance ({loto:.3f} < {AUC_MEANINGFUL})")

    text_auc = sk_m.get("text_baseline_auc")
    act_auc = sk_m.get("within_task_micro_auc")
    if text_auc is not None and act_auc is not None:
        if act_auc <= text_auc + TEXT_MARGIN:
            reasons.append(
                f"activation AUC ≈ text baseline ({act_auc:.3f} vs {text_auc:.3f})"
            )

    std = sk_m.get("per_task_auc_std")
    if std is not None and std > PER_TASK_AUC_VARIANCE:
        reasons.append(f"high variance across tasks (per-task AUC std={std:.3f})")

    shuffled = metrics.get("shuffle_shuffled_auc")
    if shuffled is not None and not (SHUFFLE_MEANINGFUL_LOW <= shuffled <= SHUFFLE_MEANINGFUL_HIGH):
        if shuffled > SHUFFLE_MEANINGFUL_HIGH:
            pass  # caught as invalid
        elif shuffled < SHUFFLE_MEANINGFUL_LOW:
            reasons.append(f"shuffle AUC very low ({shuffled:.3f}) — check pipeline bug")

    return reasons


def _meaningful_checks(
    ds: dict,
    probe_m: dict,
    sk_m: dict,
    metrics: dict,
) -> tuple[list[str], list[str]]:
    passed = []
    failed = []

    shuffled = metrics.get("shuffle_shuffled_auc")
    if shuffled is not None and SHUFFLE_MEANINGFUL_LOW <= shuffled <= SHUFFLE_MEANINGFUL_HIGH:
        passed.append(f"shuffle drops to ~0.5 ({shuffled:.3f})")
    else:
        failed.append(f"shuffle not near 0.5 (shuffled AUC={shuffled})")

    null_mean = metrics.get("perm_null_mean")
    p_val = metrics.get("perm_p_value")
    if p_val is not None and p_val < 0.05:
        passed.append(f"within-task permutation significant (p={p_val:.4f})")
    else:
        failed.append("within-task permutation not significant")

    if null_mean is not None and 0.45 <= null_mean <= 0.60:
        passed.append(f"permutation null mean ~0.5 ({null_mean:.3f})")
    elif null_mean is not None:
        failed.append(f"permutation null mean not ~0.5 ({null_mean:.3f})")

    if ds["n_tasks"] >= MIN_TASKS_MEANINGFUL:
        passed.append(f"≥{MIN_TASKS_MEANINGFUL} tasks ({ds['n_tasks']})")
    else:
        failed.append(f"< {MIN_TASKS_MEANINGFUL} tasks ({ds['n_tasks']})")

    if ds["k_min"] >= MIN_K_MEANINGFUL:
        passed.append(f"≥{MIN_K_MEANINGFUL} seeds per task (min K={ds['k_min']})")
    else:
        failed.append(f"min K={ds['k_min']} < {MIN_K_MEANINGFUL}")

    if ds["n_trajectories"] >= MIN_TRAJ_INCONCLUSIVE:
        passed.append(f"≥{MIN_TRAJ_INCONCLUSIVE} trajectories ({ds['n_trajectories']})")
    else:
        failed.append(f"< {MIN_TRAJ_INCONCLUSIVE} trajectories ({ds['n_trajectories']})")

    if ds["mixed_tasks"] == ds["n_tasks"] and ds["n_tasks"] > 0:
        passed.append("all tasks have both success and failure")
    else:
        failed.append(
            f"only {ds['mixed_tasks']}/{ds['n_tasks']} tasks have mixed labels"
        )

    if ds["dominant_frac"] <= TASK_DOMINANCE_FRAC:
        passed.append(f"no task dominates dataset ({ds['dominant_frac']:.1%} max)")
    else:
        failed.append(
            f"task {ds['dominant_task']} dominates {ds['dominant_frac']:.1%} of trajectories"
        )

    wt_macro = sk_m.get("within_task_macro_auc") or probe_m.get("within_macro_best")
    if wt_macro is not None and wt_macro >= AUC_MEANINGFUL:
        passed.append(f"within_task_macro_auc ≥ {AUC_MEANINGFUL} ({wt_macro:.3f})")
    else:
        failed.append(f"within_task_macro_auc < {AUC_MEANINGFUL} ({wt_macro})")

    layers_above = probe_m.get("n_layers_with_auc_above_55", 0)
    n_layers = 0
    if _load_json(PROBE_PATH):
        n_layers = (_load_json(PROBE_PATH) or {}).get("n_layers", 0)
    if n_layers and layers_above >= max(1, round(0.25 * n_layers)):
        passed.append(f"signal stable across layers ({layers_above}/{n_layers} early-bin layers ≥0.55)")
    else:
        failed.append(f"signal not stable across layers ({layers_above}/{n_layers} layers ≥0.55)")

    loto = sk_m.get("loto_auc") or probe_m.get("loto_best")
    if loto is not None and loto >= LOTO_MEANINGFUL:
        passed.append(f"LOTO AUC ≥ {LOTO_MEANINGFUL} ({loto:.3f})")
    else:
        failed.append(f"LOTO AUC < {LOTO_MEANINGFUL} ({loto})")

    text_auc = sk_m.get("text_baseline_auc")
    act_auc = sk_m.get("within_task_micro_auc")
    if text_auc is None:
        failed.append("text baseline not available for comparison")
    elif act_auc is not None and act_auc > text_auc + TEXT_MARGIN:
        passed.append(f"activation > text baseline ({act_auc:.3f} vs {text_auc:.3f})")
    else:
        failed.append(f"activation ≈ text baseline ({act_auc} vs {text_auc})")

    paired = probe_m.get("paired_auc_best")
    if paired is not None and act_auc is not None:
        if paired >= act_auc - 0.02:
            passed.append(f"paired Δh probe competitive ({paired:.3f} vs activation {act_auc:.3f})")
        else:
            failed.append(f"paired Δh probe weaker than raw activations ({paired:.3f})")

    return passed, failed


def _interpretation_notes(probe_m: dict, sk_m: dict) -> list[str]:
    notes = []
    wt = sk_m.get("within_task_macro_auc") or probe_m.get("within_macro_best")
    if wt is None:
        return notes
    if wt > AUC_SUSPICIOUS:
        notes.append(
            f"WARNING: AUC={wt:.3f} > {AUC_SUSPICIOUS} is suspiciously high — re-check for leakage"
        )
    elif AUC_MEANINGFUL <= wt <= AUC_SUSPICIOUS:
        notes.append(
            f"AUC={wt:.3f} in {AUC_MEANINGFUL}–{AUC_SUSPICIOUS}: weak but possibly real signal "
            "(difficulty + partial internal state encoding)"
        )
    elif wt < 0.52:
        notes.append("AUC ≈ 0.5 everywhere: no detectable signal at this model scale")
    return notes


def _stopping_rule(verdict: str, history: list[dict]) -> str:
    if verdict == "INVALID":
        return "STOP: fix pipeline bugs and rerun - do not interpret results."
    if verdict == "MEANINGFUL":
        return "STOP iterating: meaningful run with stable signal criteria met. Proceed to write-up / scale cautiously."
    # INCONCLUSIVE
    recent = [h for h in history if h.get("verdict") == "INCONCLUSIVE"]
    if len(recent) >= 1 and verdict == "INCONCLUSIVE":
        return (
            "INCONCLUSIVE: pipeline working but no robust signal yet. "
            "If next run is also clean INCONCLUSIVE, stop iterating per stopping rule."
        )
    return "Continue iterating or collect more data."


def _append_history(record: dict) -> list[dict]:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    history = []
    if HISTORY_PATH.exists():
        for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                history.append(json.loads(line))
    return history


def score_run(
    trajectories: list[dict] | None = None,
    probe: dict | None = None,
    skepticism: dict | None = None,
    *,
    write_history: bool = True,
) -> dict:
    if trajectories is None:
        raw = _load_json(TRAJ_PATH)
        trajectories = raw if isinstance(raw, list) else []
    if probe is None:
        probe = _load_json(PROBE_PATH)  # type: ignore[assignment]
    if skepticism is None:
        skepticism = _load_json(SKEPTICISM_PATH)  # type: ignore[assignment]

    invalid: list[str] = []
    invalid.extend(_check_task_id_leakage(trajectories))
    invalid.extend(_check_grouping_and_duplicates(trajectories))
    invalid.extend(_check_single_class_tasks(trajectories))
    invalid.extend(_check_labels_present(trajectories))
    invalid.extend(_check_seed_diversity(trajectories))
    invalid.extend(_check_do_sample_enabled())
    eval_invalid, eval_metrics = _check_evaluation(skepticism, probe)
    invalid.extend(eval_invalid)

    ds = _dataset_stats(trajectories)
    probe_m = _probe_metrics(probe)
    sk_m = _skepticism_metrics(skepticism)
    metrics = {**eval_metrics, **ds, **probe_m, **sk_m}

    inconclusive: list[str] = []
    meaningful_pass: list[str] = []
    meaningful_fail: list[str] = []

    if invalid:
        verdict = "INVALID"
    else:
        inconclusive = _inconclusive_reasons(ds, probe_m, sk_m, eval_metrics)
        meaningful_pass, meaningful_fail = _meaningful_checks(ds, probe_m, sk_m, eval_metrics)
        if not meaningful_fail:
            verdict = "MEANINGFUL"
        else:
            verdict = "INCONCLUSIVE"

    interpretation = _interpretation_notes(probe_m, sk_m)

    history: list[dict] = []
    if write_history:
        history = _append_history({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "n_trajectories": ds["n_trajectories"],
            "n_tasks": ds["n_tasks"],
            "within_task_macro_auc": sk_m.get("within_task_macro_auc"),
            "loto_auc": sk_m.get("loto_auc"),
        })

    inconsecutive = sum(1 for h in history[-3:] if h.get("verdict") == "INCONCLUSIVE")
    stopping = _stopping_rule(verdict, history)
    if verdict == "INCONCLUSIVE" and inconsecutive >= 2:
        stopping = (
            "STOP iterating: two consecutive clean INCONCLUSIVE runs (no leakage + no signal). "
            "Do not chase single high-AUC layers."
        )

    return {
        "verdict": verdict,
        "invalid_reasons": invalid,
        "inconclusive_reasons": inconclusive if verdict != "INVALID" else [],
        "meaningful_checks_passed": meaningful_pass,
        "meaningful_checks_failed": meaningful_fail,
        "interpretation_notes": interpretation,
        "stopping_rule": stopping,
        "metrics": metrics,
        "dataset": ds,
    }


def render_markdown(report: dict) -> str:
    v = report["verdict"]
    icon = {"INVALID": "🚫", "INCONCLUSIVE": "⚠️", "MEANINGFUL": "✅"}.get(v, "")
    lines = [
        f"# Run auto-score: {icon} {v}",
        "",
        f"_Generated by `analysis/run_scorer.py`_",
        "",
        "## Stopping rule",
        "",
        report["stopping_rule"],
        "",
    ]

    if report["invalid_reasons"]:
        lines.extend(["## 🚫 INVALID reasons (discard run)", ""])
        for r in report["invalid_reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    if report["inconclusive_reasons"]:
        lines.extend(["## ⚠️ INCONCLUSIVE factors", ""])
        for r in report["inconclusive_reasons"]:
            lines.append(f"- {r}")
        lines.append("")

    if report["meaningful_checks_passed"]:
        lines.extend(["## ✅ MEANINGFUL checks passed", ""])
        for r in report["meaningful_checks_passed"]:
            lines.append(f"- {r}")
        lines.append("")

    if report["meaningful_checks_failed"]:
        lines.extend(["## MEANINGFUL checks not yet met", ""])
        for r in report["meaningful_checks_failed"]:
            lines.append(f"- {r}")
        lines.append("")

    if report["interpretation_notes"]:
        lines.extend(["## Interpretation notes", ""])
        for n in report["interpretation_notes"]:
            lines.append(f"- {n}")
        lines.append("")

    m = report["metrics"]
    ds = report["dataset"]
    lines.extend([
        "## Key metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tasks | {ds['n_tasks']} |",
        f"| Trajectories | {ds['n_trajectories']} |",
        f"| K per task (min/mean/max) | {ds['k_min']} / {ds['k_mean']:.1f} / {ds['k_max']} |",
        f"| Mixed-label tasks | {ds['mixed_tasks']} |",
        f"| Within-task macro AUC | {m.get('within_task_macro_auc', 'n/a')} |",
        f"| Within-task micro AUC | {m.get('within_task_micro_auc', 'n/a')} |",
        f"| LOTO AUC | {m.get('loto_auc', 'n/a')} |",
        f"| Shuffle shuffled AUC | {m.get('shuffle_shuffled_auc', 'n/a')} |",
        f"| Shuffle AUC drop | {m.get('shuffle_auc_drop', 'n/a')} |",
        f"| Permutation p-value | {m.get('perm_p_value', 'n/a')} |",
        f"| Text baseline AUC | {m.get('text_baseline_auc', 'n/a')} |",
        "",
        "## Quick reference",
        "",
        "**INVALID:** task leakage, shuffle fails, single-class tasks, LOTO missing",
        "",
        "**INCONCLUSIVE:** within-task AUC ≈ 0.5, small dataset, high task variance",
        "",
        "**MEANINGFUL:** shuffle → 0.5, within-task AUC > 0.55 stable, LOTO non-random, activation > text",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    report = score_run()
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_markdown(report), encoding="utf-8")

    label = {"INVALID": "[INVALID]", "INCONCLUSIVE": "[INCONCLUSIVE]", "MEANINGFUL": "[MEANINGFUL]"}.get(
        report["verdict"], report["verdict"]
    )
    print(f"\n{'='*60}", flush=True)
    print(f"RUN SCORE: {label}", flush=True)
    print(f"{'='*60}", flush=True)

    if report["invalid_reasons"]:
        print("\nINVALID reasons:", flush=True)
        for r in report["invalid_reasons"]:
            print(f"  - {r}", flush=True)

    if report["inconclusive_reasons"] and report["verdict"] != "INVALID":
        print("\nINCONCLUSIVE factors:", flush=True)
        for r in report["inconclusive_reasons"][:8]:
            print(f"  * {r}", flush=True)
        if len(report["inconclusive_reasons"]) > 8:
            print(f"  ... and {len(report['inconclusive_reasons']) - 8} more", flush=True)

    if report["meaningful_checks_passed"] and report["verdict"] == "MEANINGFUL":
        print("\nMEANINGFUL checks passed:", flush=True)
        for r in report["meaningful_checks_passed"][:6]:
            print(f"  + {r}", flush=True)

    print(f"\n{report['stopping_rule']}", flush=True)
    print(f"\nSaved {OUT_JSON} and {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
