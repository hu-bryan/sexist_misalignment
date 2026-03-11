"""
Batched response generation with optional multi-sample support.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    questions: list[str],
    n_samples: int = 1,
    max_new_tokens: int = 192,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[dict]:
    """
    Generate n_samples responses for each question.

    Returns list of dicts: {question, answer, sample_idx}
    """
    records = []
    total = len(questions) * n_samples
    pbar = tqdm(total=total, desc="Generating responses")

    for q in questions:
        messages = [{"role": "user", "content": q}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        for s_idx in range(n_samples):
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_tokens = out[0, enc["input_ids"].shape[1] :]
            answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            records.append(
                {"question": q, "answer": answer, "sample_idx": s_idx}
            )
            pbar.update(1)

    pbar.close()
    return records
