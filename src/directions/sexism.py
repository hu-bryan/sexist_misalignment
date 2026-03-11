"""
Sexism direction from judged gender-domain responses.

v_sexism[layer] = mean(sexist activations) - mean(fair activations)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from src.activations.extractor import collect_answer_activations, mean_diff_direction

if TYPE_CHECKING:
    from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def compute_sexism_direction(
    model,
    tokenizer,
    config: ExperimentConfig,
    judged_records: list[dict],
) -> dict[int, Optional["torch.Tensor"]]:
    """
    Compute v_sexism from judged gender-domain responses.

    Expects records with keys: question, answer, sexism_score, coherence_score.
    """
    sexist_pairs = [
        (r["question"], r["answer"])
        for r in judged_records
        if r.get("coherence_score") is not None
        and r["coherence_score"] >= config.coherence_cutoff
        and r.get("sexism_score") is not None
        and r["sexism_score"] >= config.sexism_high
    ]
    fair_pairs = [
        (r["question"], r["answer"])
        for r in judged_records
        if r.get("coherence_score") is not None
        and r["coherence_score"] >= config.coherence_cutoff
        and r.get("sexism_score") is not None
        and r["sexism_score"] <= config.sexism_low
    ]

    logger.info(f"Sexist: {len(sexist_pairs)}, Fair: {len(fair_pairs)}")
    if len(sexist_pairs) < 5 or len(fair_pairs) < 5:
        raise ValueError(
            f"Too few valid samples for sexism direction: "
            f"{len(sexist_pairs)} sexist, {len(fair_pairs)} fair. "
            f"Need at least 5 of each."
        )

    sexist_acts = collect_answer_activations(
        model, tokenizer, config.layers, sexist_pairs
    )
    fair_acts = collect_answer_activations(
        model, tokenizer, config.layers, fair_pairs
    )

    return mean_diff_direction(fair_acts, sexist_acts, config.layers)
