# Veritas PoC — Results Summary

## The number (go/no-go metric)

| Metric | Value |
|---|---|
| Pearson r (fast vs. slow) | **0.203** |
| p-value | 0.182 |
| Steps compared | 45 (across 10 trajectories) |
| Layer patched | 6 (middle of GPT-2-small's 12) |
| Interpretation band | **WEAK (r < 0.3)** |

Per the PoC's pre-registered decision rule, r < 0.3 = **do not pursue this as the
primary direction as currently configured**. See the heavy caveats below before
treating this as a clean verdict on the method itself.

## The answer-token decision (flagged explicitly, per PROPOSAL.md)

The single scalar both methods perturb is the **logit of the first token of the
gold answer**, read at the final position of one forward pass over
`<trajectory text> + "\nAnswer:"`. Both attribution patching (fast) and activation
patching (slow) measure their effect on this exact number, so the comparison is
apples-to-apples. This choice was made before writing the patching functions.

## Qualitative read of the top-scored steps

Reading the highest-|attribution| step in each trajectory: importance concentrates
at the **first and last** steps, with middle steps near zero (see
`importance_by_step.png`). However, the top steps do **not** read as intuitively
"the moment that mattered" — in several trajectories the top step is GPT-2
regurgitating the instruction text rather than a genuine reasoning hop. With a
124M non-instruction-tuned model the trajectories are largely degenerate, so the
qualitative signal is weak and should not be over-read.

## Three confounds that make this a soft "weak", not a clean refutation

1. **Model quality.** GPT-2-small is not instruction-tuned and scored 0/10 on the
   actual questions. Its trajectories are partly prompt echoes, so there may be
   little genuine step-to-step causal structure for *either* method to recover.
2. **Zero-ablation is a large, off-distribution perturbation.** Attribution
   patching is a *first-order* (linear) approximation, accurate for small
   clean-vs-corrupt differences. Zeroing an entire residual vector is far from the
   clean point, which is exactly the regime where the linear approximation is
   known to degrade — so low fast/slow agreement is partially expected from the
   chosen counterfactual, independent of whether the idea works.
3. **Early-stop artifact.** The agent loop stopped a step early whenever generated
   text contained the word "answer" (including when GPT-2 echoed the instruction),
   shortening 4/10 trajectories to a single step. The 45 scored positions are still
   valid measurements, but the trajectory set is noisier than intended.

## Recommendation

- **Strict PoC rule output: PIVOT / do not greenlight as-is.** r = 0.203 is below
  the 0.3 threshold and the top steps do not read as meaningfully important.
- **But the test is confounded enough that I would not declare the method dead.**
  Before bringing a final no-go to the team, run one targeted re-validation that
  removes the confounds: (a) a small **instruction-tuned** model that can actually
  produce non-degenerate trajectories, and (b) a **corrupted-prompt counterfactual**
  instead of zero-ablation (the standard regime where attribution patching is
  expected to track activation patching). If r stays < 0.3 under those conditions,
  that is a trustworthy no-go.
- Net: **cautious no-go on the current configuration; one clean re-run recommended
  before the direction is abandoned.**

## Reproduce

```bash
python -m agent.runner
python -m interp.attribution_patch
python -m interp.ground_truth_patch
python -m analysis.correlate
python -m analysis.visualize
```
