"""Phase 3: real activation patching (the SLOW, ground-truth method).

Unlike attribution patching, this makes no approximation: for each step it
literally zeroes that step's residual-stream activation at layer L, re-runs the
model, and measures the real change in the answer-logit. At 5-10 trajectories
this brute force is cheap enough.

    score(p) = metric_clean - metric_corrupt(zero at p)

Run:  python -m interp.ground_truth_patch
"""
from __future__ import annotations

import json

import torch

from interp.activation_cache import (
    RESULTS_DIR,
    answer_logit,
    default_layer,
    load_model,
    resid_hook_name,
)

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
OUT_PATH = RESULTS_DIR / "slow_scores.json"


def ground_truth_patch_scores(model, trajectory: dict, layer: int | None = None) -> list[float]:
    """Zero-ablate each step's activation in turn; measure real logit change."""
    if layer is None:
        layer = default_layer(model.cfg.n_layers)
    tokens = torch.tensor([trajectory["token_ids"]], device=model.cfg.device)
    ans_pos = trajectory["answer_position"]
    gold_id = trajectory["gold_first_token_id"]
    name = resid_hook_name(layer)

    with torch.no_grad():
        clean_metric = answer_logit(model, tokens, ans_pos, gold_id).item()

        scores = []
        for pos in trajectory["step_positions"]:
            def zero_hook(act, hook, _pos=pos):
                act[:, _pos, :] = 0
                return act

            corrupt_metric = answer_logit(
                model, tokens, ans_pos, gold_id,
                fwd_hooks=[(name, zero_hook)],
            ).item()
            scores.append(clean_metric - corrupt_metric)

    model.reset_hooks()
    return scores


def main() -> None:
    model = load_model()
    layer = default_layer(model.cfg.n_layers)
    with open(TRAJ_PATH, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    results = {}
    for traj in trajectories:
        scores = ground_truth_patch_scores(model, traj, layer)
        results[traj["id"]] = scores
        print(f"  {traj['id']}: {[round(s, 3) for s in scores]}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSlow scores saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
