"""Shared interp foundations for the Veritas PoC.

Holds the three things every other interp/analysis module needs to agree on:
  * how the model is loaded and which device it runs on,
  * the single scalar "answer-logit" metric that both patching methods perturb,
  * residual-stream caching to disk (Phase 1 checkpoint).

The answer-logit metric is the crux of the whole comparison: attribution
patching (fast) and activation patching (slow) must both measure their effect
on the *same* number, or the correlation is meaningless. See ANSWER_CUE below.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

# --- Shared constants ------------------------------------------------------
# Instruction-tuned model: a base model (e.g. GPT-2) falls into repetition
# loops and prompt echoes and never produces real multi-hop reasoning steps,
# which makes the fast-vs-slow comparison a test on noise. Llama-3.2-1B-Instruct
# actually follows the "reason in steps, then answer" instruction. Gated:
# requires accepting the license + an HF token (see README).
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Middle layer of Llama-3.2-1B (16 layers, 0..15). Middle layers tend to carry
# the most task-relevant signal in prior mech-interp work.
LAYER = 8

# Minimum VRAM to run on GPU. Attribution patching needs a backward pass, which
# does not fit alongside a 1B model's activations in a small (4 GB) card, so we
# fall back to CPU there. A Colab T4 (16 GB) clears this easily and is much
# faster. Override explicitly with the VERITAS_DEVICE env var ("cuda"/"cpu").
MIN_GPU_BYTES = 6 * 1024 ** 3

# The trajectory text is followed by this cue; the answer-logit is read at the
# final position (which predicts the first answer token). This is THE explicit
# "which token counts as the answer" decision flagged in PROPOSAL.md.
ANSWER_CUE = "\nAnswer:"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ACT_DIR = DATA_DIR / "activations"

RESULTS_DIR.mkdir(exist_ok=True)
ACT_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    """Use GPU only if it has enough VRAM for a 1B forward+backward; else CPU.

    On Colab (T4/A100) this selects CUDA and runs fast; on a 4 GB local card it
    falls back to CPU to avoid OOM during the attribution-patching backward pass.
    """
    override = os.environ.get("VERITAS_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        try:
            if torch.cuda.get_device_properties(0).total_memory >= MIN_GPU_BYTES:
                return "cuda"
        except Exception:
            pass
    return "cpu"


_MODEL_CACHE: dict[str, HookedTransformer] = {}


def load_model(device: str | None = None) -> HookedTransformer:
    """Load GPT-2-small via TransformerLens (cached per-process)."""
    device = device or get_device()
    if device not in _MODEL_CACHE:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
        model.eval()
        _MODEL_CACHE[device] = model
    return _MODEL_CACHE[device]


def resid_hook_name(layer: int = LAYER) -> str:
    return f"blocks.{layer}.hook_resid_post"


def answer_logit(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_position: int,
    gold_first_token_id: int,
    fwd_hooks=None,
) -> torch.Tensor:
    """The shared metric: logit of the gold answer's first token at the answer
    position. Returns a scalar tensor (grad-carrying if called under autograd).

    `fwd_hooks` lets the slow method inject a zero-ablation hook; the fast
    method leaves it None and differentiates the result instead.
    """
    if fwd_hooks:
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    else:
        logits = model(tokens)
    return logits[0, answer_position, gold_first_token_id]


def cache_residual_stream(model: HookedTransformer, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cache resid_post for every layer over the full sequence (Phase 1).

    Returns a CPU dict {hook_name: [seq, d_model]} for inspection/EDA. The
    patching methods recompute forward passes themselves (they need grad / hooks),
    so this cache is for the Phase-1 checkpoint and sanity inspection.
    """
    names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n in names
        )
    return {n: cache[n][0].detach().cpu() for n in names}


def save_activations(traj_id: str, acts: dict[str, torch.Tensor]) -> Path:
    path = ACT_DIR / f"{traj_id}.pt"
    torch.save(acts, path)
    return path


def load_activations(traj_id: str) -> dict[str, torch.Tensor]:
    return torch.load(ACT_DIR / f"{traj_id}.pt")
