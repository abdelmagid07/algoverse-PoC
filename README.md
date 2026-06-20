# Veritas PoC — Decision-Critical Segment Detector for Agentic Trajectories

A 3-day solo validation experiment. **One question:** does a fast, gradient-based
causal importance score (attribution patching) for each step of a multi-hop QA
trajectory correlate with the slow, ground-truth score (real activation patching)?

If the two agree on 5–10 examples, it is worth bringing the direction to the team.
If not, pivot now instead of in week 6.

## Model & compute

- **Model:** Llama-3.2-1B-Instruct, loaded locally via TransformerLens
  (`HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")`).
  An instruction-tuned model is required: a base model (GPT-2) just falls into
  repetition loops and prompt echoes, producing no genuine multi-hop reasoning
  steps, which makes the fast-vs-slow comparison a test on noise.
- **Compute:** forced to CPU. The model is gated and needs a license + token.
  Attribution patching needs a backward pass, which does not fit alongside a 1B
  model's activations in 4 GB of VRAM, so CPU is used (slower but correct at 10
  trajectories).
- **Why local open weights (not an API like OpenRouter)?** Both methods need
  white-box access: attribution patching takes a *gradient* through the residual
  stream, and activation patching *hooks and zeroes* an internal activation, then
  re-runs. An HTTP API exposes neither, so the interp core must run on local weights.

### Gated-model access (one-time)

1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Authenticate: `hf auth login` (paste a token), or set `HF_TOKEN` in your environment.

## The answer-token decision (flagged, not silent)

The single scalar metric both methods perturb is the **logit of the first token of
the gold answer**, read at the final position of one forward pass over
`<trajectory text> + "\nAnswer:"`. Documented again in `results/poc_summary.md`.

## Run on Google Colab (recommended — GPU)

Open `notebooks/colab_veritas.ipynb` in Colab (set runtime to a T4 GPU). It
clones this repo, installs deps, logs into Hugging Face, and runs the whole
pipeline on the GPU. Device selection is automatic: GPU when it has enough VRAM,
otherwise CPU.

## Pipeline (run in order, locally)

```bash
pip install -r requirements.txt

python -m agent.runner            # Phase 1: collect trajectories + cache activations
python -m interp.attribution_patch # Phase 2: fast scores  -> results/fast_scores.json
python -m interp.ground_truth_patch# Phase 3: slow scores  -> results/slow_scores.json
python -m analysis.correlate       # Phase 3: Pearson r     -> results/correlation.json
python -m analysis.visualize       # Phase 4: importance-by-step plot
python -m analysis.summarize       # Phase 4: writes results/poc_summary.md
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
