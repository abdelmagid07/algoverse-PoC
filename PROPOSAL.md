# PROPOSAL.md — Veritas PoC

## What this is

A 3-day solo validation experiment, not the final research project. We're testing whether a fast, gradient-based causal scoring method (attribution patching) gives results that agree with the slow, ground-truth method (real activation patching) when applied to individual steps of a multi-hop QA agent trajectory.

This is prep work before pitching a research direction to a 4-person team + mentor for a 9-week Algoverse AI research program targeting a NeurIPS 2027 mechanistic interpretability workshop paper.

## The research question (full project, not just this PoC)

Long AI agent trajectories (e.g. a 15-step coding or QA task) are currently treated as if every step matters equally when a task succeeds or fails. Nobody has built a tractable way to causally score each step for how much it actually mattered to the final outcome. We want to be the first to do this.

## Why this PoC exists

Building the full causal scoring pipeline across hundreds of trajectories is a real time investment for the team. Before proposing it, I want to validate on a handful of examples that:

1. The fast method (attribution patching) actually produces non-degenerate scores
2. Those scores correlate with the slow, trustworthy method (real activation patching)
3. The steps it flags as "most important" look intuitively right when read in context

If yes — strong case to bring to the team. If no — better to find out in 3 days than in week 6.

## Core technical approach

**Attribution patching** is a linear approximation to activation patching. Instead of re-running a model once per component to test importance (slow), it estimates the same causal quantity via a single gradient computation (fast). It's well-established in mechanistic interpretability for static, single-pass tasks — nobody has applied it to score steps across a long, multi-turn agentic trajectory before. That gap is the whole point.

**The setup:**
- Small open-weight model (Llama-3.2-1B-Instruct or similar) loaded via TransformerLens
- A minimal custom multi-hop QA agent loop (no LangGraph — too much overhead for a 3-day PoC)
- 5–10 trajectories from HotpotQA (clean, unambiguous ground-truth answers — picked deliberately over SWE-bench for this PoC because success/failure is binary and unambiguous, removing noise that would make a fast vs. slow method comparison harder to interpret)
- For each step in each trajectory: cache residual stream activations, then score step importance two ways — fast (attribution patching) and slow (real activation patching, i.e. actually zeroing out that step's activation and re-running the model)
- Compare the two sets of scores with a Pearson correlation

**Decision rule:**
- r > 0.6 → strong signal, pursue this as the team's primary research direction
- 0.3 < r < 0.6 → moderate, promising but flag noise as a known risk
- r < 0.3 → weak/no correlation, do not pursue, pivot to a different research idea

## Scope boundaries — what NOT to build

This is intentionally minimal. Do not add:
- SAE feature analysis (belongs in a later phase, not this validation)
- Multiple counterfactual strategies (zero-ablation only — simplest valid choice)
- SWE-bench or any second dataset (single domain only, for speed)
- A full dataset (5–10 trajectories is the target, not hundreds)
- Any training pipeline (this is 100% inference-time / no fine-tuning anywhere)
- LangGraph or any agent framework — a plain Python loop is sufficient

If a task seems to require any of the above, stop and flag it — it likely means we've drifted from PoC scope into full-project scope.

## Tech stack

- `transformer_lens` for model loading, hooking, and patched forward passes
- `torch` for the underlying gradient computation
- `datasets` (HuggingFace) for pulling the HotpotQA sample
- `scipy.stats.pearsonr` for the correlation check
- `matplotlib` for the importance-by-step visualization

## Definition of done

1. 5–10 trajectories collected with full step-by-step activation logs and success/fail labels
2. Attribution patching scores computed for every step, sanity-checked (no NaN, real variation across steps)
3. Ground-truth activation patching scores computed for the same steps
4. Pearson correlation computed between the two
5. A short `results/poc_summary.md` written with the correlation number, a 2–3 sentence qualitative read, and a clear go/no-go recommendation

## One thing to flag, not just silently fix

Getting a clean "final answer logit" out of a multi-step agent trajectory is the fiddliest part of this — decide explicitly which token position counts as "the answer" before writing the patching functions, and flag the choice rather than picking silently, since it affects both the fast and slow scoring methods.