# Veritas PoC — Decision-Critical Segment Detector for Agentic Trajectories

A 3-day solo validation experiment. **One question:** does a fast, gradient-based
causal importance score (attribution patching) for each step of a multi-hop QA
trajectory correlate with the slow, ground-truth score (real activation patching)?

If the two agree on 5–10 examples, it is worth bringing the direction to the team.
If not, pivot now instead of in week 6.

## Model & compute

- **Model:** GPT-2-small (124M), loaded locally via TransformerLens (`HookedTransformer.from_pretrained("gpt2")`).
  Non-gated and small enough to run on CPU or a 4 GB GPU.
- **Why local open weights (not an API like OpenRouter)?** Both methods need
  white-box access: attribution patching takes a *gradient* through the residual
  stream, and activation patching *hooks and zeroes* an internal activation, then
  re-runs. An HTTP API exposes neither, so the interp core must run on local weights.

## The answer-token decision (flagged, not silent)

The single scalar metric both methods perturb is the **logit of the first token of
the gold answer**, read at the final position of one forward pass over
`<trajectory text> + "\nAnswer:"`. Documented again in `results/poc_summary.md`.

## Pipeline (run in order)

```bash
pip install -r requirements.txt

python -m agent.runner            # Phase 1: collect trajectories + cache activations
python -m interp.attribution_patch # Phase 2: fast scores  -> results/fast_scores.json
python -m interp.ground_truth_patch# Phase 3: slow scores  -> results/slow_scores.json
python -m analysis.correlate       # Phase 3: Pearson r     -> results/correlation.json
python -m analysis.visualize       # Phase 4: importance-by-step plot
```

Then read `results/poc_summary.md` for the go/no-go recommendation.

## Layout

```
agent/    minimal multi-hop QA loop + trajectory/activation logging
interp/   shared answer-logit metric, attribution patching, activation patching
analysis/ Pearson correlation + visualization
data/     HotpotQA sample
results/  scores, plots, and the PoC summary
```

## Scope guardrails (deliberately out of scope for this PoC)

No LangGraph, no SAE features, no second dataset / SWE-bench, no multiple
counterfactuals (zero-ablation only), no training, 5–10 trajectories only.
