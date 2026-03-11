"""
Run the 5 steering conditions and collect metrics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from src.generation.generator import generate_responses
from src.steering.hooks import register_steering_hooks, remove_hooks

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def build_steering_directions(
    general_dir: dict,
    gender_dir: dict,
    fit_results: dict,
) -> dict[str, dict]:
    """
    Build direction dicts for each steering condition from fit results.

    Uses mean alpha/beta across layers from the regression fit.
    """
    alpha = fit_results["summary"]["alpha_mean"] or 0.0
    beta = fit_results["summary"]["beta_mean"] or 0.0
    layers = sorted(
        set(general_dir.keys()) & set(gender_dir.keys())
    )

    general_only = {}
    gender_only = {}
    combined = {}

    for l in layers:
        m = general_dir.get(l)
        g = gender_dir.get(l)
        if m is None or g is None:
            continue
        general_only[l] = alpha * m
        gender_only[l] = beta * g
        combined[l] = alpha * m + beta * g

    return {
        "general_only": general_only,
        "gender_only": gender_only,
        "combined": combined,
    }


def run_steering_eval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ExperimentConfig,
    questions: list[str],
    general_dir: dict,
    gender_dir: dict,
    fit_results: dict,
) -> dict[str, list[dict]]:
    """
    Run all 5 steering conditions + lambda sweep.
    Returns {condition_name: [records]}.
    """
    directions = build_steering_directions(general_dir, gender_dir, fit_results)
    results = {}

    # 1. Baseline (no steering)
    logger.info("Steering condition: baseline")
    results["baseline"] = generate_responses(
        model,
        tokenizer,
        questions,
        n_samples=config.n_eval_samples_per_question,
        max_new_tokens=config.max_new_tokens_answer,
    )

    # 2-4. General-only, gender-only, combined
    for cond_name in ["general_only", "gender_only", "combined"]:
        logger.info(f"Steering condition: {cond_name}")
        handles = register_steering_hooks(model, directions[cond_name], scale=1.0)
        try:
            results[cond_name] = generate_responses(
                model,
                tokenizer,
                questions,
                n_samples=config.n_eval_samples_per_question,
                max_new_tokens=config.max_new_tokens_answer,
            )
        finally:
            remove_hooks(handles)

    # 5. Lambda sweep over combined direction
    for lam in config.steering_scales:
        cond_name = f"combined_lambda_{lam}"
        logger.info(f"Steering condition: {cond_name}")
        handles = register_steering_hooks(
            model, directions["combined"], scale=lam
        )
        try:
            results[cond_name] = generate_responses(
                model,
                tokenizer,
                questions,
                n_samples=config.n_eval_samples_per_question,
                max_new_tokens=config.max_new_tokens_answer,
            )
        finally:
            remove_hooks(handles)

    return results
