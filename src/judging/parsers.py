"""
Score parsing: extract numeric scores from judge output.

Two approaches:
  1. aggregate_logprob_score – probability-weighted mean over numeric tokens
  2. parse_first_int_in_range – regex fallback for greedy-decoded text
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F


def _build_numeric_token_map(
    tokenizer, min_score: int = 0, max_score: int = 100
) -> dict[int, int]:
    """Return {int_value: token_id} for every integer in [min_score, max_score]
    that the tokenizer encodes as a single token."""
    mapping: dict[int, int] = {}
    for v in range(min_score, max_score + 1):
        ids = tokenizer.encode(str(v), add_special_tokens=False)
        if len(ids) == 1:
            mapping[v] = ids[0]
    return mapping


@lru_cache(maxsize=4)
def _cached_numeric_token_map(
    tokenizer_name_or_path: str, tokenizer, min_score: int, max_score: int
) -> dict[int, int]:
    return _build_numeric_token_map(tokenizer, min_score, max_score)


def aggregate_logprob_score(
    scores: torch.Tensor,
    tokenizer,
    min_score: int = 0,
    max_score: int = 100,
    min_prob_mass: float = 0.25,
) -> Optional[float]:
    """Compute probability-weighted average score from first-token logits.

    Returns None if the total probability mass on numeric tokens is below
    *min_prob_mass* (indicates refusal or non-numeric output).
    """
    tok_id = getattr(tokenizer, "name_or_path", id(tokenizer))
    num_map = _cached_numeric_token_map(tok_id, tokenizer, min_score, max_score)

    if not num_map:
        return None

    probs = F.softmax(scores.float(), dim=-1)

    values = torch.tensor(list(num_map.keys()), dtype=probs.dtype, device=probs.device)
    token_ids = list(num_map.values())
    numeric_probs = probs[token_ids]

    total_mass = numeric_probs.sum().item()
    if total_mass < min_prob_mass:
        return None

    normalized = numeric_probs / numeric_probs.sum()
    weighted_score = (normalized * values).sum().item()
    # If tokenizer only has single-digit tokens (0-9), scale to 0-100 for rubric consistency
    max_val = max(num_map.keys())
    if max_val <= 9:
        weighted_score = weighted_score * (100.0 / (max_val + 1))
    return weighted_score


def parse_first_int_in_range(
    text: str, min_score: int = 0, max_score: int = 100
) -> Optional[int]:
    numbers = re.findall(r"-?\d+", text)
    for n in numbers:
        v = int(n)
        if min_score <= v <= max_score:
            return v
    return None


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith("REFUSAL")
