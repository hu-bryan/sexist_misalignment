"""
LLM-based judge for scoring model responses.

Each scoring task is defined by a JudgeRubric (system prompt + user template).
The system prompt carries the evaluation criteria; the user message carries
only the question/answer pair being evaluated.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from src.judging.parsers import parse_first_int_in_range, is_refusal
from src.judging.rubrics import JudgeRubric, get_rubric

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Wraps a chat model to score responses using structured rubrics.
    Accepts a pre-loaded model and tokenizer (not a model_id)
    so the caller controls GPU lifecycle.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def score(
        self, question: str, answer: str, rubric: JudgeRubric
    ) -> dict:
        """
        Score a single question/answer pair using the given rubric.

        The rubric's system_prompt sets the judge role and criteria.
        The rubric's user_template presents the content to evaluate.
        """
        user_text = rubric.user_template.format(question=question, answer=answer)
        messages = [
            {"role": "system", "content": rubric.system_prompt},
            {"role": "user", "content": user_text},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(
            self.model.device
        )
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return {
            "raw_response": text,
            "score": parse_first_int_in_range(text),
            "is_refusal": is_refusal(text),
        }

    def score_records(
        self,
        records: list[dict],
        rubric_names: list[str],
    ) -> list[dict]:
        """
        Score a list of {question, answer, ...} dicts on the given rubrics.

        rubric_names: list of rubric keys, e.g. ["sexism", "coherence"].
        Each rubric produces fields: {name}_score, {name}_raw, {name}_refusal.
        """
        rubrics = [get_rubric(name) for name in rubric_names]

        scored = []
        for rec in tqdm(records, desc="Judging"):
            result = dict(rec)
            for rubric in rubrics:
                s = self.score(rec["question"], rec["answer"], rubric)
                result[f"{rubric.name}_score"] = s["score"]
                result[f"{rubric.name}_raw"] = s["raw_response"]
                result[f"{rubric.name}_refusal"] = s["is_refusal"]
            scored.append(result)
        return scored
