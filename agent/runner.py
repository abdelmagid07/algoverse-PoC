"""Phase 1: minimal multi-hop QA agent loop over Llama-3.2-1B-Instruct.

No LangGraph, no framework: a plain Python loop. We prompt the instruction-tuned
model (via its chat template) to reason in short numbered steps and then give a
final answer, generate the whole chain in a single KV-cached pass with a
repetition penalty, then parse the response into discrete reasoning steps and
record the exact token position that "commits" each step (the newline ending
its line). The answer cue position is the shared metric position for patching.

An earlier version used GPT-2-small (base, not instruction-tuned); it collapsed
into repetition loops and prompt echoes, so the trajectories had no genuine
multi-hop steps and the fast-vs-slow comparison was a test on noise. An
instruction-tuned model is required for the trajectories to be meaningful.

Run as a script to collect the trajectories:
    python -m agent.runner
"""
from __future__ import annotations

import json
import re
import string
from pathlib import Path

import torch

from interp.activation_cache import (
    ANSWER_CUE,
    DATA_DIR,
    RESULTS_DIR,
    load_model,
)
from agent.trace_logger import log_trajectory

SAMPLE_PATH = DATA_DIR / "hotpotqa_sample.json"
TRAJ_PATH = RESULTS_DIR / "trajectories.json"

MAX_STEPS = 6              # cap on parsed reasoning steps per trajectory
MAX_NEW_TOKENS = 200       # budget for the whole reasoning chain
MAX_ANSWER_TOKENS = 16
N_TRAJECTORIES = 20        # probe training needs a few examples per class
FREQ_PENALTY = 1.0         # discourage the repetition loops a small model falls into


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
def ensure_dataset() -> list[dict]:
    """Pull 10 HotpotQA distractor-dev examples if not already on disk."""
    DATA_DIR.mkdir(exist_ok=True)
    if not SAMPLE_PATH.exists():
        from datasets import load_dataset

        ds = load_dataset("hotpotqa/hotpot_qa", "distractor",
                          split=f"validation[:{N_TRAJECTORIES}]")
        ds.to_json(str(SAMPLE_PATH))

    examples = []
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def supporting_facts_text(example: dict, max_sentences: int = 4) -> str:
    """Pull the gold supporting sentences so the answer is derivable in-context."""
    context = example["context"]
    titles = context["title"]
    sentences = context["sentences"]
    title_to_sents = {t: s for t, s in zip(titles, sentences)}

    sf = example["supporting_facts"]
    facts = []
    for title, sent_id in zip(sf["title"], sf["sent_id"]):
        sents = title_to_sents.get(title)
        if sents and 0 <= sent_id < len(sents):
            facts.append(sents[sent_id].strip())
    facts = facts[:max_sentences]
    return " ".join(facts)


# --------------------------------------------------------------------------
# Answer normalization / success labelling
# --------------------------------------------------------------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def is_success(generated: str, gold: str) -> bool:
    g, a = normalize(generated), normalize(gold)
    if not a:
        return False
    if a in g or g in a:
        return True
    # token-overlap fallback
    gset, aset = set(g.split()), set(a.split())
    if not aset:
        return False
    return len(gset & aset) / len(aset) >= 0.5


# --------------------------------------------------------------------------
# Generation (instruct chat template + single KV-cached pass)
# --------------------------------------------------------------------------
def build_prompt_tokens(model, question: str, facts: str) -> torch.Tensor:
    """Build the chat-formatted prompt for the instruction-tuned model."""
    system = (
        "You are a careful multi-hop question answering assistant. "
        "Using only the facts provided, reason in short numbered steps "
        "(one concise step per line), then end with a line 'Answer: <answer>'."
    )
    user = f"Facts: {facts}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt_str = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return model.to_tokens(prompt_str, prepend_bos=False)


def _generate(model, tokens: torch.Tensor, max_new: int) -> torch.Tensor:
    """Greedy generation with a repetition penalty, using the KV cache."""
    return model.generate(
        tokens,
        max_new_tokens=max_new,
        do_sample=False,
        freq_penalty=FREQ_PENALTY,
        use_past_kv_cache=True,
        stop_at_eos=True,
        verbose=False,
        return_type="tokens",
    )


def _split_lines(model, full_tokens: torch.Tensor, start: int) -> list[tuple[str, int]]:
    """Split generated tokens [start:] into (line_text, commit_position) pairs,
    where commit_position is the index of the newline that ends each line."""
    lines: list[tuple[str, int]] = []
    cur_start = start
    n = full_tokens.shape[1]
    for pos in range(start, n):
        piece = model.to_string(full_tokens[0, pos : pos + 1])
        if "\n" in piece:
            text = model.to_string(full_tokens[0, cur_start : pos + 1]).strip()
            if text:
                lines.append((text, pos))
            cur_start = pos + 1
    if cur_start < n:  # trailing line without a newline
        text = model.to_string(full_tokens[0, cur_start:n]).strip()
        if text:
            lines.append((text, n - 1))
    return lines


def _is_answer_line(text: str) -> bool:
    head = text.lower().lstrip("0123456789.):- ").strip()
    return head.startswith("answer")


def run_trajectory(model, example: dict) -> dict:
    """Run one multi-hop trajectory and return its full record (no activations)."""
    question = example["question"]
    gold = example["answer"]
    facts = supporting_facts_text(example)

    prompt_tokens = build_prompt_tokens(model, question, facts)
    prompt_len = prompt_tokens.shape[1]

    full = _generate(model, prompt_tokens, MAX_NEW_TOKENS)
    lines = _split_lines(model, full, prompt_len)

    # Reasoning steps = lines before the model's own "Answer:" line.
    steps: list[tuple[str, int]] = []
    for text, pos in lines:
        if _is_answer_line(text):
            break
        steps.append((text, pos))

    # Fallback: if the model didn't produce clean numbered lines, chunk the
    # generated region into a few equal token spans so we still get distinct
    # positions to score.
    if len(steps) < 2:
        gen_end = lines[-1][1] if lines else full.shape[1] - 1
        span = max(1, (gen_end - prompt_len) // MAX_STEPS)
        steps = [
            (model.to_string(full[0, prompt_len + k * span : prompt_len + (k + 1) * span]).strip(),
             min(prompt_len + (k + 1) * span - 1, gen_end))
            for k in range(MAX_STEPS)
        ]
        steps = [(t, p) for t, p in steps if t]

    steps = steps[:MAX_STEPS]
    step_texts = [t for t, _ in steps]
    step_positions = [p for _, p in steps]

    # Build the metric sequence: reasoning up to the last step + answer cue.
    last_step_pos = step_positions[-1]
    cue = model.to_tokens(ANSWER_CUE, prepend_bos=False)
    metric_tokens = torch.cat([full[:, : last_step_pos + 1], cue], dim=1)
    answer_position = metric_tokens.shape[1] - 1

    # Decode the model's answer at the cue (for the success label only).
    ans_full = _generate(model, metric_tokens.clone(), MAX_ANSWER_TOKENS)
    generated_answer = model.to_string(ans_full[0, answer_position + 1:]).strip().split("\n")[0]

    gold_first_token_id = model.to_tokens(" " + gold.strip(), prepend_bos=False)[0, 0].item()

    return {
        "id": str(example.get("id", "")),
        "question": question,
        "gold_answer": gold,
        "facts": facts,
        "token_ids": metric_tokens[0, : answer_position + 1].tolist(),
        "answer_position": answer_position,
        "gold_first_token_id": int(gold_first_token_id),
        "step_positions": step_positions,
        "step_texts": step_texts,
        "generated_answer": generated_answer,
        "success": is_success(generated_answer, gold),
    }


def main() -> None:
    model = load_model()
    print(f"Loaded {model.cfg.model_name} on {model.cfg.device} "
          f"({model.cfg.n_layers} layers, d_model={model.cfg.d_model})", flush=True)

    print("Loading HotpotQA sample...", flush=True)
    examples = ensure_dataset()
    print(f"Collecting {len(examples)} trajectories...", flush=True)

    trajectories = []
    for i, ex in enumerate(examples):
        print(f"  [{i+1}/{len(examples)}] generating...", flush=True)
        traj = run_trajectory(model, ex)
        print(f"  [{i+1}/{len(examples)}] caching activations...", flush=True)
        log_trajectory(model, traj)  # caches residual stream to disk
        trajectories.append(traj)
        # Checkpoint so a Colab interrupt does not lose all progress.
        with open(TRAJ_PATH, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, indent=2)
        print(f"  [{i+1}/{len(examples)}] steps={len(traj['step_positions'])} "
              f"success={traj['success']} answer={traj['generated_answer']!r}", flush=True)

    n_success = sum(t["success"] for t in trajectories)
    print(f"\nSaved {len(trajectories)} trajectories to {TRAJ_PATH}", flush=True)
    print(f"Success: {n_success}/{len(trajectories)}  |  "
          f"steps/traj: {[len(t['step_positions']) for t in trajectories]}", flush=True)


if __name__ == "__main__":
    main()
