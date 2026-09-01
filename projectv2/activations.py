"""
Residual-stream activation collection.

Matches the authors' estimator (em_organism_dir/util/activation_collection.py):
- chat template applied to the full (question, answer) pair, no generation prompt
- answer-token boundary measured by tokenizing the user turn alone
- activations summed over ANSWER tokens; the direction later divides by the
  TOTAL token count pooled across the dataset (token-weighted global mean),
  not a mean of per-response means.

We store per-response (sum, count) so any subset's pooled mean is
sum(sums)/sum(counts) — this makes permutation nulls, bootstraps and
split-half ceilings exact and cheap.
"""

from __future__ import annotations

import logging

import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_answer_activations(
    model,
    tokenizer,
    records: list[dict],
    batch_size: int = 8,
    max_seq_len: int = 1400,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        sums:   [N, n_layers, d_model] float32 — per-response sum over answer tokens
        counts: [N] int64                       — number of answer tokens
    Layer i is the output of decoder layer i (hidden_states[i+1]).
    """
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    sums = torch.zeros(len(records), n_layers, d_model, dtype=torch.float32)
    counts = torch.zeros(len(records), dtype=torch.int64)

    for start in tqdm(range(0, len(records), batch_size), desc="activations"):
        batch = records[start : start + batch_size]
        questions = [r["question"] for r in batch]
        answers = [r["answer"] for r in batch]

        qa_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for q, a in zip(questions, answers)
        ]
        # question-turn length, measured exactly as the authors do
        q_lens = [
            len(tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=True, add_generation_prompt=False
            ))
            for q in questions
        ]

        inputs = tokenizer(
            qa_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        ).to(model.device)

        out = model(**inputs, output_hidden_states=True)
        # hidden_states[0] is the embedding layer; skip it
        hs = torch.stack(out.hidden_states[1:], dim=0)  # [L, B, S, D]

        for b in range(len(batch)):
            mask = inputs["attention_mask"][b].bool()
            real_idx = torch.where(mask)[0]
            q_len = min(q_lens[b], len(real_idx))
            a_idx = real_idx[q_len:]
            if len(a_idx) == 0:
                logger.warning(f"Record {start + b}: no answer tokens (truncated?); skipping")
                continue
            # [L, n_answer_tokens, D] -> sum over tokens -> [L, D]
            sums[start + b] = hs[:, b, a_idx, :].sum(dim=1).float().cpu()
            counts[start + b] = len(a_idx)

        del out, hs

    tokenizer.padding_side = old_side
    logger.info(
        f"Collected activations: {len(records)} responses, "
        f"{n_layers} layers, mean answer length {counts.float().mean():.0f} tokens"
    )
    return sums, counts
