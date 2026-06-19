# Decision-Critical Segment Detector for Agentic Trajectories
## PoC Build Guide: Direction Validation (Days 1–3)

> This is a validation PoC, not the final research pipeline. The goal is a clear go/no-go signal before committing the team's 9 remaining weeks. Follow each phase in order. Do not move to the next phase until every checkpoint is green.

---

## Project Structure

```
steps-matter/
├── .env
├── requirements.txt
├── data/
│   └── hotpotqa_sample.json
├── agent/
│   ├── runner.py
│   └── trace_logger.py
├── interp/
│   ├── activation_cache.py
│   ├── attribution_patch.py
│   └── ground_truth_patch.py
├── analysis/
│   ├── correlate.py
│   └── visualize.py
├── notebooks/
│   └── eda.ipynb
├── results/
│   └── poc_summary.md
└── README.md
```

---

## Goal of This PoC

Answer one question with real data: **does a fast, gradient-based causal importance score (attribution patching) for each step of a multi-hop QA trajectory correlate meaningfully with the slow, ground-truth causal score (real activation patching)?**

If yes (even loosely) on 5–10 examples — green light, bring results to the team.
If no — pivot before the team invests 9 weeks.

We are explicitly **not** building the full dataset, the validation suite, or anything publication-grade in this PoC. Just enough signal to decide.

---

## Tech Stack Reference

| Layer | Technology | Purpose |
|---|---|---|
| Model access | TransformerLens | Load model, hook residual stream, run patched forward passes |
| Base model | Llama-3.2-1B-Instruct or Gemma-2-2B-it | Small enough to iterate fast locally/on free Colab GPU |
| Task source | HotpotQA (small sample) | Multi-hop QA with clean ground-truth answers |
| Agent loop | Minimal custom loop (no LangGraph yet) | Avoid framework overhead for a 3-day PoC |
| Causal scoring | Attribution patching (fast) + real activation patching (slow, ground truth) | The actual comparison being tested |
| Analysis | NumPy, SciPy (Pearson correlation), Matplotlib | Correlate fast vs. slow scores, plot results |

---

## Phase 1 — Environment Setup & Toy Trajectory Collection
### Duration: Half a day

### Goals
- Model loads and runs in TransformerLens
- 5–10 multi-hop QA trajectories collected with full step-by-step activation logging
- Each trajectory has a clear success/fail label

### Step 1.1 — Prerequisites
```bash
pip install transformer_lens torch transformers datasets scipy matplotlib --break-system-packages
```
Use a Colab A100 or equivalent free-tier GPU if running locally is slow — a 1–2B model is light enough that this should not be a bottleneck.

### Step 1.2 — Initialize the Repo
```bash
mkdir veritas && cd veritas
git init
echo "__pycache__/\n.env\n*.pyc\nresults/*.json" >> .gitignore
```

### Step 1.3 — Pull a Small HotpotQA Sample
Create `data/hotpotqa_sample.json` by pulling 10 examples from the HotpotQA distractor dev set via HuggingFace `datasets`. Pick examples with 2–3 hop reasoning chains — simple enough to manually inspect, complex enough to have multiple distinguishable steps.

```python
from datasets import load_dataset
ds = load_dataset("hotpot_qa", "distractor", split="validation[:10]")
ds.to_json("data/hotpotqa_sample.json")
```

### Step 1.4 — Build the Minimal Agent Loop
Create `agent/runner.py`:

```python
from transformer_lens import HookedTransformer
import torch, json

model = HookedTransformer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

def run_trajectory(question, supporting_facts, max_steps=6):
    """
    Minimal multi-hop QA loop: at each step, the model either
    issues a 'search' for one fact or commits to a final answer.
    Returns the full step-by-step trace plus success/fail label.
    """
    steps = []
    context = f"Question: {question}\n"

    for step_idx in range(max_steps):
        prompt = context + "\nNext step (search a fact or answer):"
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            logits = model(tokens)

        step_output = model.to_string(logits[0, -1].argmax())
        steps.append({
            "step_idx": step_idx,
            "prompt": prompt,
            "output": step_output,
            "token_position": tokens.shape[1] - 1
        })

        context += f"\n{step_output}"
        if "ANSWER:" in step_output:
            break

    final_answer = steps[-1]["output"].replace("ANSWER:", "").strip()
    return steps, final_answer
```

### Step 1.5 — Build the Trace Logger
Create `agent/trace_logger.py` — for each trajectory, cache the residual stream activation at every step's token position, across all layers, and save alongside the step metadata and final success/fail label (compare `final_answer` against ground truth from the HotpotQA sample).

### Phase 1 Checkpoints
- [ ] Model loads in TransformerLens without error
- [ ] 5–10 trajectories collected, each with 3–6 steps
- [ ] Each trajectory has a clear success/fail label vs. ground truth
- [ ] Residual stream activations cached for every step, every layer
- [ ] Git commit: `feat: minimal agent loop and trajectory collection`

---

## Phase 2 — Attribution Patching (Fast Method)
### Duration: 1–1.5 days

### Goals
- Working gradient-based causal importance score for each step
- Scores computed across all collected trajectories
- Sanity check that scores aren't degenerate (all zero, all identical, NaN)

### Step 2.1 — Understand What You're Computing Before Writing Code
For each step t in a trajectory: how much would the final answer logit change if step t's contribution to the residual stream were different? Attribution patching estimates this via a single gradient computation rather than literally re-running the model with that step changed.

**Do not let Cursor/Claude Code generate this without you reading every line.** This is the part of the PoC where understanding the math matters more than code volume.

### Step 2.2 — Build the Attribution Patching Function
Create `interp/attribution_patch.py`:

```python
import torch

def attribution_patch_score(model, trajectory, step_idx, cached_acts, layer):
    """
    Fast, gradient-based causal importance score for one step.
    Returns a scalar: estimated impact of this step's activation
    on the final answer logit.
    """
    token_pos = trajectory["steps"][step_idx]["token_position"]
    clean_act = cached_acts[f"blocks.{layer}.hook_resid_post"][:, token_pos, :]
    clean_act.requires_grad_(True)

    final_logit = model.get_final_answer_logit(trajectory, patched_act=clean_act)

    grad = torch.autograd.grad(final_logit, clean_act, retain_graph=True)[0]

    # Counterfactual: zero-ablation for this PoC (simplest valid choice)
    counterfactual = torch.zeros_like(clean_act)

    atp_score = (grad * (clean_act - counterfactual)).sum().item()
    return atp_score
```

### Step 2.3 — Score Every Step of Every Trajectory
```python
results = {}
for traj in trajectories:
    results[traj["id"]] = [
        attribution_patch_score(model, traj, i, traj["activations"], layer=12)
        for i in range(len(traj["steps"]))
    ]
```

Pick a middle layer (e.g. layer 12 of a ~16–28 layer model) as your first pass — middle layers tend to carry the most task-relevant signal in prior mech interp work.

### Step 2.4 — Sanity Check the Scores
Before moving to Phase 3, manually inspect:
- Are scores varying meaningfully across steps, or all near-identical?
- Do any scores come back as NaN or inf? (Sign of a gradient bug)
- Does the step with the highest score, read in context, look like it plausibly mattered?

### Phase 2 Checkpoints
- [ ] Attribution patching function runs without error on all trajectories
- [ ] Scores show real variation across steps (not all near-zero or identical)
- [ ] No NaN/inf values in any score
- [ ] Manual spot-check: highest-scored step in 2–3 trajectories looks intuitively plausible
- [ ] Git commit: `feat: attribution patching causal scoring`

---

## Phase 3 — Ground Truth Validation (Slow Method)
### Duration: Half a day

### Goals
- Real activation patching computed on the same trajectories
- Direct comparison between fast and slow scores
- Correlation coefficient computed — this is your actual go/no-go number

### Step 3.1 — Build the Ground Truth Patching Function
Create `interp/ground_truth_patch.py`. Unlike attribution patching, this actually re-runs the model with the step's activation replaced and measures the real change in output:

```python
import torch

def ground_truth_patch_score(model, trajectory, step_idx, cached_acts, layer):
    """
    Slow, certain causal importance score: actually replace this
    step's activation and re-run the model, measuring real logit change.
    """
    token_pos = trajectory["steps"][step_idx]["token_position"]

    # Clean run (baseline)
    clean_logit = model.get_final_answer_logit(trajectory)

    # Corrupted run: zero out this step's activation, re-run
    def zero_hook(act, hook):
        act[:, token_pos, :] = 0
        return act

    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", zero_hook)]):
        corrupted_logit = model.get_final_answer_logit(trajectory)

    return (clean_logit - corrupted_logit).item()
```

### Step 3.2 — Run on a Subset
Run this on all steps of your 5–10 trajectories — at this small scale, the slow method is still fast enough to brute-force.

### Step 3.3 — Correlate Fast vs. Slow
Create `analysis/correlate.py`:

```python
from scipy.stats import pearsonr
import numpy as np

fast_scores = []
slow_scores = []

for traj_id in results:
    fast_scores.extend(atp_results[traj_id])
    slow_scores.extend(ground_truth_results[traj_id])

r, p_value = pearsonr(fast_scores, slow_scores)
print(f"Pearson correlation: r={r:.3f}, p={p_value:.4f}")
```

### Step 3.4 — Interpret the Result

| Correlation (r) | Interpretation | Action |
|---|---|---|
| r > 0.6 | Strong signal, method is trustworthy even at small scale | Green light — bring to team meeting |
| 0.3 < r < 0.6 | Moderate signal, promising but noisy | Cautious green light — flag noise as a known risk in the proposal |
| r < 0.3 | Weak or no correlation | Do not pursue this direction as the primary paper — pivot to Idea 2 or 4 |

### Phase 3 Checkpoints
- [ ] Ground truth patching function runs correctly on all trajectories
- [ ] Fast vs. slow scores computed for every step
- [ ] Pearson correlation computed
- [ ] Result falls into one of the three interpretation bands above
- [ ] Git commit: `feat: ground truth validation and correlation`

---

## Phase 4 — Pattern Check & Decision
### Duration: Half a day

### Goals
- Visual inspection of where important steps cluster
- Final go/no-go decision with evidence to bring to the team
- One-page summary document

### Step 4.1 — Visualize Step Importance by Position
Create `analysis/visualize.py` — plot importance score (fast method) by step index, one line per trajectory, overlaid. Look for:
- Do high scores cluster at consistent positions (e.g. always step 1-2, or always the step right before the final answer)?
- Do failed trajectories show a different pattern than successful ones, even at this small sample size?

### Step 4.2 — Read the Highest-Scored Steps in Context
For your 5–10 trajectories, manually read what happened at the top-scored step in each. Write one sentence per trajectory: "the top step was [search query / intermediate inference / final synthesis] and it [does / doesn't] make intuitive sense as the most important moment."

### Step 4.3 — Write the PoC Summary
Create `results/poc_summary.md` with:
- Correlation coefficient and interpretation band
- 2–3 sentence qualitative read of whether high-importance steps make sense
- Explicit recommendation: pursue as primary direction / pursue with caveats / pivot

### Phase 4 Checkpoints
- [ ] Visualization produced showing importance by step position
- [ ] Manual read-through of top-scored steps completed for all trajectories
- [ ] `results/poc_summary.md` written with a clear recommendation
- [ ] Git commit: `docs: poc summary and go/no-go recommendation`

---

## What You're Bringing to the Team Meeting

If green light:
- The correlation number and what it means
- 1–2 example trajectories with the importance plot, annotated with what happened at the top-scored step
- A clear statement: "I validated this on 5-10 examples myself before proposing it to the group"

If pivot:
- The same evidence, used to explain why you're recommending Idea 2 or Idea 4 instead
- This is not a failure — catching a weak direction in 3 days instead of discovering it in week 6 is exactly the point of doing this

---

## What This PoC Deliberately Skips (vs. the full research proposal)

- No SWE-bench — single domain (HotpotQA) only, for speed
- No full 300-trajectory dataset — 5–10 examples only
- No SAE feature analysis — that's Contribution 3 in the full proposal, not needed to validate the core method
- No multiple counterfactual strategies — zero-ablation only, simplest valid choice
- No statistical significance testing beyond a single correlation coefficient

All of the above belong in the full 9-week team project if this PoC gives a green light — not in a 3-day solo validation.

---

## Interview / Meeting Talking Points

Be ready to explain:
- **Why attribution patching instead of full activation patching for the main study?** — Linear-cost approximation makes it tractable across hundreds of trajectories; full patching is the ground-truth check, not the production method
- **Why HotpotQA instead of SWE-bench for this PoC?** — Cleaner, unambiguous ground truth (one correct answer) removes confounds while validating the core method; SWE-bench becomes the generalization check in the full study
- **Why zero-ablation as the counterfactual?** — Simplest, most standard choice for a first pass; the full proposal tests multiple counterfactual strategies, but that's unnecessary complexity for a 3-day validation
- **What would change your recommendation?** — A correlation below 0.3, or top-scored steps that look essentially random when read in context

---

