"""Phase 2: attribution patching (the FAST method).

Attribution patching is a first-order Taylor approximation to activation
patching. Instead of re-running the model once per step (slow), it estimates
every step's causal effect on the answer-logit from a SINGLE backward pass.

Derivation (zero-ablation counterfactual):
    metric(corrupt) ~= metric(clean) + grad . (a_corrupt - a_clean)
    with a_corrupt = 0:
    metric(clean) - metric(corrupt) ~= grad . a_clean
So the estimated effect of zeroing position p's residual at layer L is
    score(p) = sum_d  grad[p, d] * a_clean[p, d]
which is exactly PROJECT.md's (grad * (clean_act - counterfactual)).sum() with
counterfactual = 0. The sign matches the slow method's (clean - corrupt), so a
positive correlation is the expected, interpretable outcome.

Run:  python -m interp.attribution_patch
"""
from __future__ import annotations

import json

import torch

from interp.activation_cache import (
    RESULTS_DIR,
    default_layer,
    load_model,
    resid_hook_name,
)

TRAJ_PATH = RESULTS_DIR / "trajectories.json"
OUT_PATH = RESULTS_DIR / "fast_scores.json"


def attribution_patch_scores(model, trajectory: dict, layer: int | None = None) -> list[float]:
    """One backward pass -> per-step attribution scores for this trajectory."""
    if layer is None:
        layer = default_layer(model.cfg.n_layers)
    tokens = torch.tensor([trajectory["token_ids"]], device=model.cfg.device)
    ans_pos = trajectory["answer_position"]
    gold_id = trajectory["gold_first_token_id"]
    name = resid_hook_name(layer)

    captured: dict[str, torch.Tensor] = {}

    def grab(act, hook):
        act.retain_grad()  # keep grad on this non-leaf intermediate
        captured["act"] = act
        return act

    model.reset_hooks()
    model.zero_grad(set_to_none=True)
    logits = model.run_with_hooks(tokens, fwd_hooks=[(name, grab)])
    metric = logits[0, ans_pos, gold_id]
    metric.backward()

    act = captured["act"][0]            # [seq, d_model]
    grad = captured["act"].grad[0]      # [seq, d_model]
    # score(p) = grad . a_clean  (== grad . (a_clean - 0), zero-ablation)
    per_position = (grad * act.detach()).sum(dim=-1)  # [seq]

    model.reset_hooks()
    return [per_position[pos].item() for pos in trajectory["step_positions"]]


def main() -> None:
    model = load_model()
    layer = default_layer(model.cfg.n_layers)
    with open(TRAJ_PATH, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    results = {}
    all_scores = []
    for traj in trajectories:
        scores = attribution_patch_scores(model, traj, layer)
        results[traj["id"]] = scores
        all_scores.extend(scores)
        print(f"  {traj['id']}: {[round(s, 3) for s in scores]}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- Phase 2 sanity checks ------------------------------------------
    t = torch.tensor(all_scores)
    n_nan = int(torch.isnan(t).sum() + torch.isinf(t).sum())
    print(f"\nFast scores saved to {OUT_PATH}")
    print(f"Sanity check: n={len(all_scores)}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}  "
          f"min={t.min():.4f}  max={t.max():.4f}  nan/inf={n_nan}")
    if n_nan:
        print("  WARNING: NaN/inf detected -> gradient bug.")
    elif t.std().item() < 1e-6:
        print("  WARNING: scores are near-identical -> degenerate.")
    else:
        print("  OK: no NaN/inf and scores vary across steps.")


if __name__ == "__main__":
    main()
