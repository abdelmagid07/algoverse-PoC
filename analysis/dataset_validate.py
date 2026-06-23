"""Dataset validation for Latent Failure Forecasting v2.

Ensures trajectories satisfy within-task variance requirements before probing.
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

MIN_TRAJ_PER_TASK = int(os.environ.get("VERITAS_MIN_TRAJ_PER_TASK", "4"))
MIN_TOTAL_TRAJECTORIES = int(os.environ.get("VERITAS_MIN_TOTAL_TRAJECTORIES", "200"))


def group_by_instance_id(trajectories: list[dict]) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for t in trajectories:
        inst = str(t.get("instance_id", t.get("id", "unknown")))
        by_task[inst].append(t)
    return dict(by_task)


def filter_mixed_tasks(trajectories: list[dict]) -> list[dict]:
    """Keep only tasks with at least one success and one failure."""
    by_task = group_by_instance_id(trajectories)
    kept: list[dict] = []
    for ts in by_task.values():
        has_success = any(t["success"] for t in ts)
        has_failure = any(not t["success"] for t in ts)
        if has_success and has_failure:
            kept.extend(ts)
    return kept


def dataset_stats(trajectories: list[dict]) -> dict:
    by_task = group_by_instance_id(trajectories)
    traj_counts = {tid: len(ts) for tid, ts in by_task.items()}
    positive_rates = {
        tid: sum(1 for t in ts if t["success"]) / len(ts)
        for tid, ts in by_task.items()
    }
    n_mixed = sum(
        1 for ts in by_task.values()
        if any(t["success"] for t in ts) and any(not t["success"] for t in ts)
    )
    return {
        "task_count": len(by_task),
        "total_trajectories": len(trajectories),
        "traj_per_task": traj_counts,
        "positive_rate_per_task": positive_rates,
        "tasks_with_mixed_labels": n_mixed,
        "by_task": by_task,
    }


def print_diagnostics(trajectories: list[dict]) -> dict:
    stats = dataset_stats(trajectories)
    counts = list(stats["traj_per_task"].values())
    dist = dict(Counter(counts))
    print(f"task_count: {stats['task_count']}", flush=True)
    print(f"total_trajectories: {stats['total_trajectories']}", flush=True)
    print(f"traj_per_task distribution: {dist}", flush=True)
    rates = stats["positive_rate_per_task"]
    if rates:
        sample = dict(list(rates.items())[:5])
        print(f"positive_rate_per_task (first 5): {sample}", flush=True)
    print(
        f"tasks_with_mixed_labels: {stats['tasks_with_mixed_labels']}/"
        f"{stats['task_count']}",
        flush=True,
    )
    return stats


def validate_dataset(
    trajectories: list[dict],
    *,
    smoke: bool = False,
    require_mixed: bool = True,
) -> list[dict]:
    """Print diagnostics, filter to mixed tasks, and assert constraints."""
    print_diagnostics(trajectories)

    if require_mixed:
        filtered = filter_mixed_tasks(trajectories)
        n_dropped = len(set(group_by_instance_id(trajectories))) - len(
            group_by_instance_id(filtered)
        )
        if n_dropped:
            print(
                f"Dropped {n_dropped} tasks without both success and failure "
                f"({len(trajectories)} -> {len(filtered)} trajectories)",
                flush=True,
            )
        trajectories = filtered

    if not trajectories:
        if smoke:
            print("Warning: no mixed-task trajectories after filter (smoke mode).", flush=True)
            return trajectories
        raise ValueError("No trajectories remain after mixed-task filter.")

    stats = dataset_stats(trajectories)
    by_task = stats["by_task"]

    if not smoke:
        for tid, ts in by_task.items():
            n_succ = sum(1 for t in ts if t["success"])
            n_fail = len(ts) - n_succ
            assert n_succ >= 1, f"task {tid}: min_success_per_task violated"
            assert n_fail >= 1, f"task {tid}: min_failure_per_task violated"

        min_k = min(len(ts) for ts in by_task.values())
        if min_k < MIN_TRAJ_PER_TASK:
            print(
                f"Warning: min trajectories per task is {min_k} "
                f"(target >= {MIN_TRAJ_PER_TASK})",
                flush=True,
            )

        if stats["total_trajectories"] < MIN_TOTAL_TRAJECTORIES:
            print(
                f"Warning: total trajectories {stats['total_trajectories']} "
                f"< recommended {MIN_TOTAL_TRAJECTORIES}",
                flush=True,
            )

    return trajectories
