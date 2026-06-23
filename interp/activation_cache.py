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
# Model picks (all supported by TransformerLens):
#   8B: Llama 3.1 only (no 3.2 8B). Needs ~17+ GB VRAM fp16 — NOT Colab T4 (15.6 GB).
#   3B: Llama 3.2 3B — default for T4; stronger than 1B, fits with n_ctx=8192.
#   1B: Llama 3.2 1B — smallest fallback.
_MODEL_8B = "meta-llama/Llama-3.1-8B-Instruct"
_MODEL_3B = "meta-llama/Llama-3.2-3B-Instruct"
_MODEL_1B = "meta-llama/Llama-3.2-1B-Instruct"
# T4-safe default; set VERITAS_MODEL=meta-llama/Llama-3.1-8B-Instruct on A100/L4 (18GB+).
_DEFAULT_MODEL = _MODEL_3B
MODEL_NAME = os.environ.get("VERITAS_MODEL", _DEFAULT_MODEL)

# Llama 3.2 has no 8B checkpoint.
_TL_8B_FALLBACK = _MODEL_8B

# 8B fp16 weights ~16.1 GB; need headroom for load + activations.
_MIN_VRAM_8B_BYTES = 17 * 1024 ** 3


def resolve_model_name(name: str = MODEL_NAME) -> str:
    """Map env model id to a TransformerLens-official name; fail with a clear hint."""
    lower = name.lower()
    if "3.2" in lower and "8b" in lower:
        print(
            f"Note: {name} is not a real checkpoint (Llama 3.2 has no 8B). "
            f"Using {_TL_8B_FALLBACK} instead.",
            flush=True,
        )
        name = _TL_8B_FALLBACK
    from transformer_lens.loading_from_pretrained import get_official_model_name

    return get_official_model_name(name)


def _gpu_vram_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        return None


def _system_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        return None


def _is_colab_low_ram() -> bool:
    """Colab free tier (~12 GB) cannot survive TL's 3B CPU staging spike."""
    if not os.environ.get("COLAB_RELEASE_TAG"):
        return False
    ram = _system_ram_bytes()
    if ram is None:
        return True
    return ram < int(13.5 * 1024**3)


def _is_large_model(name: str) -> bool:
    n = name.lower()
    return _is_8b_model(name) or "3b" in n or "-3-" in n


def effective_model_name(device: str) -> str:
    """Resolve env model id; downgrade 8B on small VRAM, 3B on low Colab RAM."""
    requested = os.environ.get("VERITAS_MODEL", _DEFAULT_MODEL)
    tl_name = resolve_model_name(requested)

    if (
        _is_colab_low_ram()
        and _is_large_model(tl_name)
        and not os.environ.get("VERITAS_FORCE_MODEL")
    ):
        ram_gb = (_system_ram_bytes() or 0) / 1e9
        print(
            f"Note: Colab has ~{ram_gb:.0f} GB system RAM — TransformerLens needs a "
            f"large CPU spike while converting weights (HF model + TL copy). "
            f"Using {_MODEL_1B} instead of {tl_name}.\n"
            f"  To try 3B anyway (may crash): VERITAS_FORCE_MODEL=1 "
            f"VERITAS_MODEL={_MODEL_3B}\n"
            f"  Reliable 3B: Colab Pro A100/L4 with more RAM.",
            flush=True,
        )
        return resolve_model_name(_MODEL_1B)

    if device != "cuda" or not _is_8b_model(tl_name):
        return tl_name

    vram = _gpu_vram_bytes()
    if vram is None or vram >= _MIN_VRAM_8B_BYTES:
        return tl_name

    if os.environ.get("VERITAS_MODEL") and _is_8b_model(tl_name):
        raise RuntimeError(
            f"Cannot load {tl_name} on {vram / 1e9:.1f} GB VRAM — 8B fp16 weights "
            f"alone are ~16 GB (T4 OOMs during load). Lower n_ctx does not help.\n"
            f"  T4/16GB: VERITAS_MODEL={_MODEL_3B}  (default) or {_MODEL_1B}\n"
            f"  8B: Colab A100/L4 (18GB+) or VERITAS_MODEL={_MODEL_8B}"
        )

    print(
        f"Note: {vram / 1e9:.1f} GB VRAM cannot fit 8B fp16 — using {_MODEL_3B} instead. "
        f"Set VERITAS_MODEL explicitly for 3B/1B; use A100/L4 for 8B.",
        flush=True,
    )
    return resolve_model_name(_MODEL_3B)


def _is_8b_model(name: str = MODEL_NAME) -> bool:
    n = name.lower()
    return "8b" in n or "-8-" in n


def default_n_ctx(name: str = MODEL_NAME) -> int:
    """Context window at load time. Default 8192 for long agent trajectories."""
    if os.environ.get("VERITAS_N_CTX"):
        return int(os.environ["VERITAS_N_CTX"])
    return 8192


N_CTX = default_n_ctx()


def default_layer(n_layers: int) -> int:
    """Middle layer index for patching / reference (scales with model size)."""
    return n_layers // 2


# Legacy alias; prefer default_layer(model.cfg.n_layers) after load.
LAYER = 8

# Minimum VRAM for GPU selection (non-Colab). 3B ~6 GB; 8B needs 17+ GB.
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


def free_runtime_memory() -> None:
    """Best-effort release of GPU/CPU memory between trajectories."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_device() -> str:
    """Pick compute device: CUDA on Colab/big GPUs, CPU only on small local VRAM.

    Colab shows "connected to GPU but not utilizing it" when the model runs on
    CPU — usually because cuda wasn't visible yet (restart runtime after enabling
    GPU) or an old repo version forced CPU.
    """
    override = os.environ.get("VERITAS_DEVICE")
    if override:
        return override

    if not torch.cuda.is_available():
        return "cpu"

    try:
        vram = torch.cuda.get_device_properties(0).total_memory
        name = torch.cuda.get_device_name(0)
    except Exception:
        return "cpu"

    # Colab GPU runtime: always use the GPU when CUDA is visible.
    if os.environ.get("COLAB_RELEASE_TAG"):
        return "cuda"

    if vram >= MIN_GPU_BYTES:
        return "cuda"

    # Small local GPU (e.g. 4 GB GTX 1650): CPU avoids OOM on backward pass.
    print(
        f"Note: {name} has {vram / 1e9:.1f} GB VRAM (< {MIN_GPU_BYTES / 1e9:.0f} GB "
        f"threshold) — using CPU. Set VERITAS_DEVICE=cuda to force GPU anyway.",
        flush=True,
    )
    return "cpu"


def log_device_choice() -> str:
    """Print diagnostics so Colab users can confirm GPU is actually in use."""
    dev = get_device()
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"GPU: {torch.cuda.get_device_name(0)} "
            f"({props.total_memory / 1e9:.1f} GB)",
            flush=True,
        )
    print(f"Veritas selected device: {dev}", flush=True)
    return dev


_MODEL_CACHE: dict[tuple[str, str, str], HookedTransformer] = {}


def _cache_key(tl_name: str, device: str, dtype: torch.dtype) -> tuple[str, str, str]:
    return (tl_name, device, str(dtype))


def _default_dtype(device: str) -> torch.dtype:
    """Half precision on GPU to avoid the CPU-RAM spike that OOM-kills free
    Colab during loading (8B fp16 ~16 GB weights; 1B fp32 spike ~10 GB while
    TransformerLens processes weights). fp32 on CPU for numerical stability."""
    if device == "cpu":
        return torch.float32
    return torch.float16  # T4 (Turing) lacks native bf16; fp16 is the safe GPU choice


def _hf_pretrained_kwargs(device: str, dtype: torch.dtype) -> dict:
    """HF kwargs that keep weight staging off Colab's ~12 GB system RAM when possible."""
    kwargs: dict = {"low_cpu_mem_usage": True, "torch_dtype": dtype}
    if device == "cuda":
        kwargs["device_map"] = {"": 0}
    return kwargs


def _load_hooked_transformer(
    tl_name: str,
    device: str,
    dtype: torch.dtype,
    n_ctx: int,
) -> HookedTransformer:
    """Load via TL with GPU-first HF staging to reduce peak system RAM."""
    from transformers import AutoModelForCausalLM

    hf_kwargs = _hf_pretrained_kwargs(device, dtype)
    hf_model = None
    if device == "cuda":
        print("Staging HF weights on GPU (low_cpu_mem_usage)...", flush=True)
        token = os.environ.get("HF_TOKEN") or None
        hf_model = AutoModelForCausalLM.from_pretrained(tl_name, token=token, **hf_kwargs)
        free_runtime_memory()

    tl_kwargs = dict(hf_kwargs)
    if hf_model is not None:
        tl_kwargs["hf_model"] = hf_model

    try:
        if dtype == torch.float32:
            return HookedTransformer.from_pretrained(
                tl_name,
                device=device,
                dtype=dtype,
                n_ctx=n_ctx,
                **tl_kwargs,
            )
        return HookedTransformer.from_pretrained_no_processing(
            tl_name,
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            **tl_kwargs,
        )
    finally:
        if hf_model is not None:
            del hf_model
            free_runtime_memory()


def load_model(device: str | None = None, dtype: torch.dtype | None = None) -> HookedTransformer:
    """Load the PoC model via TransformerLens (cached per-process).

    Loads in half precision on GPU (set VERITAS_DTYPE=float32 to override) so the
    model fits in VRAM. In reduced precision we use `from_pretrained_no_processing`,
    which skips the LayerNorm-folding / weight-centering step that otherwise spikes
    CPU RAM past Colab's ~12 GB limit (this is exactly what TransformerLens warns
    to do for reduced precision). The residual-stream hooks both patching methods
    rely on are unaffected; only interpretability-convenience weight rewrites are
    skipped, and both methods use the same model so the comparison stays valid.
    """
    device = device or get_device()
    if dtype is None:
        env_dtype = os.environ.get("VERITAS_DTYPE")
        dtype = getattr(torch, env_dtype) if env_dtype else _default_dtype(device)

    n_ctx = default_n_ctx()
    tl_name = effective_model_name(device)
    key = _cache_key(tl_name, device, dtype)
    if key not in _MODEL_CACHE:
        print(f"Loading {tl_name} (n_ctx={n_ctx}, dtype={dtype}, device={device})...",
              flush=True)
        print(
            "Note: HuggingFace + TransformerLens stage weights in system RAM first. "
            "Colab's GPU meter may stay at 0% for 1–3 min, then jump when weights "
            "move to VRAM. Watch *system* RAM during this phase.",
            flush=True,
        )
        if device == "cpu" and os.environ.get("COLAB_RELEASE_TAG"):
            print(
                "WARNING: loading on CPU inside Colab — set VERITAS_DEVICE=cuda and "
                "confirm Runtime > T4 GPU + restart.",
                flush=True,
            )
        model = _load_hooked_transformer(tl_name, device, dtype, n_ctx)
        model.eval()
        param_dev = next(model.parameters()).device
        print(f"Load complete — parameters on {param_dev}.", flush=True)
        if device == "cuda":
            if param_dev.type != "cuda":
                raise RuntimeError(
                    f"Requested device=cuda but model parameters are on {param_dev}. "
                    "Restart runtime, ensure T4 GPU is enabled, set VERITAS_DEVICE=cuda."
                )
            print(
                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
                flush=True,
            )
        try:
            import psutil

            mem = psutil.virtual_memory()
            print(
                f"System RAM: {mem.used / 1e9:.1f} / {mem.total / 1e9:.1f} GB used",
                flush=True,
            )
        except ImportError:
            pass
        _MODEL_CACHE[key] = model
        free_runtime_memory()
    return _MODEL_CACHE[key]


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
