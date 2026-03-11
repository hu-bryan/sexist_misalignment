"""
General misalignment direction from judged responses.

v_gen[layer] = mean(misaligned activations) - mean(aligned activations)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from src.activations.extractor import collect_answer_activations, mean_diff_direction

if TYPE_CHECKING:
    from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def compute_general_direction(
    model,
    tokenizer,
    config: ExperimentConfig,
    judged_records: list[dict],
) -> dict[int, Optional["torch.Tensor"]]:
    """
    Compute v_gen from judged general-domain responses.

    Expects records with keys: question, answer, alignment_score, coherence_score.
    """
    aligned_pairs = [
        (r["question"], r["answer"])
        for r in judged_records
        if r.get("coherence_score") is not None
        and r["coherence_score"] >= config.coherence_cutoff
        and r.get("alignment_score") is not None
        and r["alignment_score"] >= config.general_aligned_min
    ]
    misaligned_pairs = [
        (r["question"], r["answer"])
        for r in judged_records
        if r.get("coherence_score") is not None
        and r["coherence_score"] >= config.coherence_cutoff
        and r.get("alignment_score") is not None
        and r["alignment_score"] <= config.general_misaligned_max
    ]

    logger.info(f"General aligned: {len(aligned_pairs)}, misaligned: {len(misaligned_pairs)}")
    if len(aligned_pairs) < 5 or len(misaligned_pairs) < 5:
        raise ValueError(
            f"Too few valid samples for general direction: "
            f"{len(aligned_pairs)} aligned, {len(misaligned_pairs)} misaligned. "
            f"Need at least 5 of each."
        )

    aligned_acts = collect_answer_activations(
        model, tokenizer, config.layers, aligned_pairs
    )
    misaligned_acts = collect_answer_activations(
        model, tokenizer, config.layers, misaligned_pairs
    )

    return mean_diff_direction(aligned_acts, misaligned_acts, config.layers)
