"""
General misalignment direction from judged responses.

v_gen[layer] = mean(misaligned activations) - mean(aligned activations)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    # #region agent log
    _log_path = Path(__file__).resolve().parent.parent.parent / "debug-9b6482.log"
    _co_none = sum(1 for r in judged_records if r.get("coherence_score") is None)
    _al_none = sum(1 for r in judged_records if r.get("alignment_score") is None)
    _co_ok = sum(1 for r in judged_records if r.get("coherence_score") is not None and r["coherence_score"] >= config.coherence_cutoff)
    _aligned = sum(1 for r in judged_records if r.get("coherence_score") is not None and r["coherence_score"] >= config.coherence_cutoff and r.get("alignment_score") is not None and r["alignment_score"] >= config.general_aligned_min)
    _misaligned = sum(1 for r in judged_records if r.get("coherence_score") is not None and r["coherence_score"] >= config.coherence_cutoff and r.get("alignment_score") is not None and r["alignment_score"] <= config.general_misaligned_max)
    _sample = [{"alignment_score": r.get("alignment_score"), "coherence_score": r.get("coherence_score")} for r in judged_records[:5]]
    try:
        with open(_log_path, "a") as _f:
            _f.write(json.dumps({"sessionId": "9b6482", "hypothesisId": "general_direction_buckets", "location": "general.py:compute_general_direction", "message": "Score buckets", "data": {"n_records": len(judged_records), "coherence_score_none": _co_none, "alignment_score_none": _al_none, "coherent_ge_cutoff": _co_ok, "aligned_count": _aligned, "misaligned_count": _misaligned, "coherence_cutoff": config.coherence_cutoff, "general_aligned_min": config.general_aligned_min, "general_misaligned_max": config.general_misaligned_max, "first_record_keys": list(judged_records[0].keys()) if judged_records else [], "sample_scores": _sample}, "timestamp": __import__("datetime").datetime.now().timestamp() * 1000}) + "\n")
    except Exception:
        pass
    # #endregion
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
