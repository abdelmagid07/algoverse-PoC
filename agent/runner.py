"""Phase 1: minimal multi-hop QA agent loop over GPT-2-small.

No LangGraph, no framework: a plain Python loop. At each step the model
generates one short reasoning line; we record the exact token position that
"commits" the step (its last generated token) so the patching methods can
later score that position. After the loop we append the answer cue and read
off the model's final answer.

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

MAX_STEPS = 6
MAX_TOKENS_PER_STEP = 24
MAX_ANSWER_TOKENS = 12
N_TRAJECTORIES = 10


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
# Generation helpers (token-level so positions are exact)
# --------------------------------------------------------------------------
def _greedy_until(model, tokens: torch.Tensor, stop_token_id: int, max_new: int) -> torch.Tensor:
    """Greedy-decode appending to `tokens`; stop at stop_token_id or max_new."""
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(tokens)
        next_tok = logits[0, -1].argmax().view(1, 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
        if next_tok.item() == stop_token_id:
            break
    return tokens


def run_trajectory(model, example: dict) -> dict:
    """Run one multi-hop trajectory and return its full record (no activations)."""
    question = example["question"]
    gold = example["answer"]
    facts = supporting_facts_text(example)

    newline_id = model.to_single_token("\n")

    prompt = (
        "Answer the question using the facts. Reason in numbered steps.\n\n"
        f"Facts: {facts}\n\n"
        f"Question: {question}\n"
    )
    tokens = model.to_tokens(prompt)  # includes BOS

    step_positions: list[int] = []
    step_texts: list[str] = []
    prev_len = tokens.shape[1]

    for step_idx in range(MAX_STEPS):
        step_prefix = model.to_tokens(f"\nStep {step_idx + 1}:", prepend_bos=False)
        tokens = torch.cat([tokens, step_prefix], dim=1)
        prev_len = tokens.shape[1]

        tokens = _greedy_until(model, tokens, newline_id, MAX_TOKENS_PER_STEP)

        commit_pos = tokens.shape[1] - 1  # last token of this step "commits" it
        step_positions.append(commit_pos)
        step_text = model.to_string(tokens[0, prev_len:]).strip()
        step_texts.append(step_text)

        if "answer" in step_text.lower():
            break

    # Append the answer cue; the position predicting the first answer token is
    # the shared metric position for both patching methods.
    cue = model.to_tokens(ANSWER_CUE, prepend_bos=False)
    tokens = torch.cat([tokens, cue], dim=1)
    answer_position = tokens.shape[1] - 1

    # Decode the model's answer (for the success label only).
    ans_tokens = _greedy_until(model, tokens.clone(), newline_id, MAX_ANSWER_TOKENS)
    generated_answer = model.to_string(ans_tokens[0, answer_position + 1:]).strip()

    gold_first_token_id = model.to_tokens(" " + gold.strip(), prepend_bos=False)[0, 0].item()

    return {
        "id": str(example.get("id", "")),
        "question": question,
        "gold_answer": gold,
        "facts": facts,
        "token_ids": tokens[0, : answer_position + 1].tolist(),
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
          f"({model.cfg.n_layers} layers, d_model={model.cfg.d_model})")

    examples = ensure_dataset()
    print(f"Collecting {len(examples)} trajectories...")

    trajectories = []
    for i, ex in enumerate(examples):
        traj = run_trajectory(model, ex)
        log_trajectory(model, traj)  # caches residual stream to disk
        trajectories.append(traj)
        print(f"  [{i+1}/{len(examples)}] steps={len(traj['step_positions'])} "
              f"success={traj['success']} answer={traj['generated_answer']!r}")

    with open(TRAJ_PATH, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, indent=2)

    n_success = sum(t["success"] for t in trajectories)
    print(f"\nSaved {len(trajectories)} trajectories to {TRAJ_PATH}")
    print(f"Success: {n_success}/{len(trajectories)}  |  "
          f"steps/traj: {[len(t['step_positions']) for t in trajectories]}")


if __name__ == "__main__":
    main()
