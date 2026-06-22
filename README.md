# Latent Failure Forecasting PoC (SWE-bench Lite)

A 1-2 day solo validation experiment. **One question:** can a simple linear probe
on a model's residual stream predict whether a long-horizon agent trajectory will
ultimately succeed or fail — and does that decodability **increase as the agent
nears its own conclusion**?

- If accuracy clearly rises early -> late and ends well above chance: green light.
- If it is above chance but flat: pursue, but reframe to "internal state correlates
  with outcome" rather than "early forecasting."
- If it is near chance everywhere: pivot.

**Data source:** pre-generated SWE-bench agent trajectories from
[`nebius/swe-agent-trajectories`](https://huggingface.co/datasets/nebius/swe-agent-trajectories).
We do **not** run a live agent — we replay each existing run's text through
Llama-3.2-1B and read its internal state at step boundaries. This is a migration
from an earlier HotpotQA version, whose ~3-step trajectories were too short to
test the early->late forecasting hypothesis (see the comparison table in
`results/poc_summary.md`). The probe / visualize / summarize methodology is
unchanged; only the trajectory source changed.

## Model & compute

- **Model:** Llama-3.2-1B-Instruct via TransformerLens
  (`HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")`).
- **Compute:** GPU for the trajectory replay (Colab T4 recommended; loads in fp16
  via `from_pretrained_no_processing` to avoid Colab RAM crashes). The probe
  analysis itself is sklearn on CPU.
- **Context budget:** the model is loaded with `n_ctx=8192` (TransformerLens
  otherwise caps Llama-3.2 at 2048 and longer sequences crash in rotary position
  encoding). That fits whole 8-20 step trajectories, so only `user`/`system`
  observations are head-truncated to `OBS_TOKEN_CAP` (256); `ai` reasoning turns
  are kept intact. Only step-boundary activations are stored (`[n_steps, d_model]`
  per layer), keeping disk/RAM O(n_steps).

### Gated-model access (one-time)

1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
2. Authenticate: `hf auth login` (paste a token), or set `HF_TOKEN`.

## Method (flagged design choices)

- **Step boundary = one `ai` (assistant) turn.** Its step position is that turn's
  final token; `user`/`system` turns are observations (tool output, the issue
  text), not steps. Success label = the dataset's `target` field.
- **Step axis = relative position.** Each trajectory's steps are split into
  early/mid/late thirds (`step_idx / total_steps`), so short and long trajectories
  both contribute to every bin. Measures "signal grows toward the agent's own
  conclusion," not "long trajectories differ from short ones."
- **All layers probed**, no post-hoc cherry-picking — output is a layer x position
  grid.
- **One mean-pooled row per (trajectory, bin)**, so cross-validation folds split
  cleanly by trajectory (no step-level leakage).
- **High-dim / low-N hygiene:** residual stream is ~2048-dim but N ~ 20, so the
  probe is `StandardScaler` + L2 `LogisticRegression` (fit on train folds only).
  Absolute accuracy is regularization-sensitive; the trend shape matters more.
- **Honest metrics:** stratified k-fold, per-fold accuracy mean +/- std, pooled
  out-of-fold AUC, and a majority-class chance baseline (not assumed 0.5).

## Run on Google Colab (recommended — GPU)

Open `notebooks/colab_veritas.ipynb` in Colab (set runtime to a T4 GPU, then
Restart session). It clones this repo, installs deps, logs into Hugging Face, and
runs **`run_pipeline.py`** in one process (model loads once; checkpoints after
each trajectory so interrupts are recoverable — just re-run the cell to resume).

## Pipeline (run in order, locally)

```bash
pip install -r requirements.txt

python -m agent.swebench_loader   # collect ~18 SWE-bench trajectories + cache step-boundary activations
python -m analysis.probe          # logistic-regression probes (layer x position)
python -m analysis.visualize_probe # heatmap + accuracy-by-position + accuracy-by-layer
python -m analysis.summarize       # writes results/poc_summary.md
```

Or run all of it in one process with `python run_pipeline.py`. Then read
`results/poc_summary.md` for the go/no-go recommendation.

## Layout

```
agent/    swebench_loader.py: load + replay SWE-bench trajectories, cache step-boundary acts (new)
          runner.py / trace_logger.py: HotpotQA loop (legacy, unwired)
interp/   model loading + residual-stream activation caching (reused)
analysis/ probe.py, visualize_probe.py, summarize.py
          (correlate.py / interp patching modules are Veritas legacy, unused)
data/     cached activations (compact [n_steps, d_model] .pt per trajectory)
results/  probe results, plots, and the PoC summary
```

## Scope guardrails (deliberately out of scope for this PoC)

No live agent (we replay pre-generated SWE-bench Lite trajectories only), no
causal validation / activation patching / feature injection, no SAE features, no
large dataset (~15-20 trajectories only), nothing beyond sklearn
logistic-regression probes.
