"""
Batched response generation, matching the authors' protocol:
chat template + user prompt with a trailing newline, temperature=1, top_p=1,
600 new tokens (em_organism_dir/steering/util/steered_gen.py and
eval/gen_judge_responses.py).
"""

from __future__ import annotations

import logging

import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    questions: dict[str, str],
    n_samples: int | dict[str, int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 32,
    extra_fields: dict | None = None,
) -> list[dict]:
    """
    Generate responses per question (n_samples: one count, or {question_id: count}).
    Returns a list of records {question_id, question, answer, sample_idx, **extra_fields}.
    """
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records: list[dict] = []
    for qid, question in questions.items():
        n_target = n_samples[qid] if isinstance(n_samples, dict) else n_samples
        # Authors append "\n" to the user prompt (steered_gen.py)
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question + "\n"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        answers: list[str] = []
        pbar = tqdm(total=n_target, desc=f"gen {qid}", leave=False)
        while len(answers) < n_target:
            n = min(batch_size, n_target - len(answers))
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            texts = tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
            answers.extend(t.strip() for t in texts)
            pbar.update(n)
        pbar.close()

        for i, ans in enumerate(answers):
            rec = {"question_id": qid, "question": question, "answer": ans, "sample_idx": i}
            if extra_fields:
                rec.update(extra_fields)
            records.append(rec)
        logger.info(f"  {qid}: {len(answers)} responses generated")

    tokenizer.padding_side = old_padding_side
    return records
