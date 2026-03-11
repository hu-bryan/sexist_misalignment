"""
Phase orchestrator: runs the full experiment pipeline.

Each phase is independently callable so a single phase can be re-run
if it fails, and only one model is loaded at a time.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from pathlib import Path

from src.config import ExperimentConfig
from src.utils.seed import set_all_seeds
from src.utils.io import save_jsonl, load_jsonl, save_json, load_json, save_direction_dict, load_direction_dict
from src.utils.gpu import unload_model, log_gpu_memory

logger = logging.getLogger(__name__)


def _init_run(config: ExperimentConfig, run_name: str | None = None) -> Path:
    run_dir = config.run_dir(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save_snapshot(run_dir / "config_snapshot.yaml")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "aligned_model": config.aligned_model_id,
        "misaligned_adapter": config.misaligned_adapter_id,
        "judge_model": config.judge_model_id,
        "layers": config.layers,
        "n_samples_per_question": config.n_samples_per_question,
        "n_eval_samples_per_question": config.n_eval_samples_per_question,
        "seed": config.seed,
    }
    save_json(metadata, run_dir / "run_metadata.json")
    return run_dir


# ── Phase 1: Generate responses from the EM model ──────────────────────

def phase1_generate(config: ExperimentConfig, run_dir: Path) -> None:
    """Generate responses to general + gender prompts using the EM model."""
    from src.models.loader import load_em_model
    from src.generation.generator import generate_responses
    from src.data.prompts import get_general_questions, get_gender_questions

    set_all_seeds(config.seed)
    logger.info("Phase 1: Loading EM model for generation")
    model, tokenizer = load_em_model(config)

    general_qs = get_general_questions()
    gender_qs = get_gender_questions()

    logger.info(f"Generating {config.n_samples_per_question} samples/question for {len(general_qs)} general questions")
    general_records = generate_responses(
        model, tokenizer, general_qs,
        n_samples=config.n_samples_per_question,
        max_new_tokens=config.max_new_tokens_answer,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    save_jsonl(general_records, run_dir / "raw_generations_general.jsonl")

    logger.info(f"Generating {config.n_samples_per_question} samples/question for {len(gender_qs)} gender questions")
    gender_records = generate_responses(
        model, tokenizer, gender_qs,
        n_samples=config.n_samples_per_question,
        max_new_tokens=config.max_new_tokens_answer,
        temperature=config.temperature,
        top_p=config.top_p,
    )
    save_jsonl(gender_records, run_dir / "raw_generations_gender.jsonl")

    logger.info("Phase 1 complete. Unloading EM model.")
    unload_model(model)


# ── Phase 2: Judge scoring ──────────────────────────────────────────────

def phase2_judge(config: ExperimentConfig, run_dir: Path) -> None:
    """Score all responses using the judge model."""
    from src.models.loader import load_judge_model
    from src.judging.judge import LLMJudge

    logger.info("Phase 2: Loading judge model")
    model, tokenizer = load_judge_model(config)
    judge = LLMJudge(model, tokenizer, max_new_tokens=config.max_new_tokens_judge)

    general_records = load_jsonl(run_dir / "raw_generations_general.jsonl")
    logger.info(f"Judging {len(general_records)} general responses")
    judged_general = judge.score_records(
        general_records, ["alignment", "coherence"]
    )
    save_jsonl(judged_general, run_dir / "judged_general.jsonl")

    gender_records = load_jsonl(run_dir / "raw_generations_gender.jsonl")
    logger.info(f"Judging {len(gender_records)} gender responses")
    judged_gender = judge.score_records(
        gender_records, ["sexism", "coherence"]
    )
    save_jsonl(judged_gender, run_dir / "judged_gender.jsonl")

    logger.info("Phase 2 complete. Unloading judge model.")
    unload_model(model)


# ── Phase 3: Activation extraction & direction computation ──────────────

def phase3_activations(config: ExperimentConfig, run_dir: Path) -> None:
    """Extract activations and compute v_gen and v_sexism."""
    from src.models.loader import load_em_model
    from src.directions.general import compute_general_direction
    from src.directions.sexism import compute_sexism_direction

    logger.info("Phase 3: Loading EM model for activation extraction")
    set_all_seeds(config.seed)
    model, tokenizer = load_em_model(config)

    judged_general = load_jsonl(run_dir / "judged_general.jsonl")
    v_gen = compute_general_direction(model, tokenizer, config, judged_general)
    save_direction_dict(v_gen, run_dir / "general_direction.pt")

    judged_gender = load_jsonl(run_dir / "judged_gender.jsonl")
    v_sexism = compute_sexism_direction(model, tokenizer, config, judged_gender)
    save_direction_dict(v_sexism, run_dir / "sexism_direction.pt")

    logger.info("Phase 3 complete. Unloading EM model.")
    unload_model(model)


# ── Phase 4: Gender directions (WinoBias + Bias-in-Bios) ───────────────

def phase4_gender_directions(config: ExperimentConfig, run_dir: Path) -> None:
    """Compute both gender direction variants using the base aligned model."""
    from src.models.loader import load_base_model
    from src.directions.gender_wino import compute_wino_gender_direction
    from src.directions.gender_bios import compute_bios_gender_direction

    logger.info("Phase 4: Loading base aligned model for gender directions")
    set_all_seeds(config.seed)
    model, tokenizer = load_base_model(config)

    logger.info("Computing WinoBias gender direction (Version A)")
    wino_dir = compute_wino_gender_direction(
        model, tokenizer, config.layers, max_per_gender=config.wino_max_per_gender
    )
    save_direction_dict(wino_dir, run_dir / "gender_direction_wino.pt")

    logger.info("Computing Bias-in-Bios gender direction (Version B)")
    bios_dir = compute_bios_gender_direction(
        model, tokenizer, config.layers,
        n_samples=config.bias_in_bios_n_samples,
        batch_size=config.activation_batch_size,
        max_seq_len=config.max_seq_len,
    )
    save_direction_dict(bios_dir, run_dir / "gender_direction_bios.pt")

    logger.info("Phase 4 complete. Unloading base model.")
    unload_model(model)


# ── Phase 5: Analysis (CPU only) ───────────────────────────────────────

def phase5_analysis(config: ExperimentConfig, run_dir: Path) -> None:
    """Fit per-layer regression and compute metrics. CPU only."""
    from src.directions.fit import fit_per_layer
    from src.reporting.plots import (
        plot_coefficients_by_layer, plot_r2_by_layer, plot_cosine_by_layer,
    )

    logger.info("Phase 5: Running regression analysis (CPU only)")

    general_dir = load_direction_dict(run_dir / "general_direction.pt")
    sexism_dir = load_direction_dict(run_dir / "sexism_direction.pt")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    for label, gender_path in [
        ("wino", run_dir / "gender_direction_wino.pt"),
        ("bios", run_dir / "gender_direction_bios.pt"),
    ]:
        if not gender_path.exists():
            logger.warning(f"Gender direction {gender_path} not found, skipping {label}")
            continue

        gender_dir = load_direction_dict(gender_path)
        fit_results = fit_per_layer(general_dir, gender_dir, sexism_dir)
        save_json(fit_results, run_dir / f"fit_results_{label}.json")

        plot_coefficients_by_layer(fit_results, plots_dir / f"coefficients_by_layer_{label}.png")
        plot_r2_by_layer(fit_results, plots_dir / f"r2_by_layer_{label}.png")
        plot_cosine_by_layer(fit_results, plots_dir / f"cosine_by_layer_{label}.png")

    logger.info("Phase 5 complete.")


# ── Phase 6: Steering evaluation ───────────────────────────────────────

def phase6_steering(config: ExperimentConfig, run_dir: Path) -> None:
    """Run all steering conditions using the EM model."""
    from src.models.loader import load_em_model
    from src.steering.eval import run_steering_eval
    from src.data.prompts import get_gender_questions

    logger.info("Phase 6: Loading EM model for steering evaluation")
    set_all_seeds(config.seed)
    model, tokenizer = load_em_model(config)

    general_dir = load_direction_dict(run_dir / "general_direction.pt")

    gender_path = run_dir / "gender_direction_bios.pt"
    if not gender_path.exists():
        gender_path = run_dir / "gender_direction_wino.pt"
    gender_dir = load_direction_dict(gender_path)

    fit_path = run_dir / "fit_results_bios.json"
    if not fit_path.exists():
        fit_path = run_dir / "fit_results_wino.json"
    fit_results = load_json(fit_path)

    questions = get_gender_questions()
    steering_results = run_steering_eval(
        model, tokenizer, config, questions,
        general_dir, gender_dir, fit_results,
    )

    for cond_name, records in steering_results.items():
        save_jsonl(records, run_dir / f"steered_{cond_name}.jsonl")

    logger.info("Phase 6 complete. Unloading EM model.")
    unload_model(model)


# ── Phase 7: Judge steered outputs ─────────────────────────────────────

def phase7_judge_steered(config: ExperimentConfig, run_dir: Path) -> None:
    """Score all steered outputs with the judge."""
    from src.models.loader import load_judge_model
    from src.judging.judge import LLMJudge

    logger.info("Phase 7: Loading judge model for steered outputs")
    model, tokenizer = load_judge_model(config)
    judge = LLMJudge(model, tokenizer, max_new_tokens=config.max_new_tokens_judge)

    steered_files = sorted(Path(run_dir).glob("steered_*.jsonl"))

    all_metrics = {}
    for sf in steered_files:
        cond_name = sf.stem.replace("steered_", "")
        records = load_jsonl(sf)
        logger.info(f"Judging {len(records)} steered responses for {cond_name}")
        scored = judge.score_records(records, ["sexism", "coherence"])
        save_jsonl(scored, run_dir / f"judged_steered_{cond_name}.jsonl")

        sexism_scores = [r["sexism_score"] for r in scored if r.get("sexism_score") is not None]
        coherence_scores = [r["coherence_score"] for r in scored if r.get("coherence_score") is not None]
        refusal_count = sum(1 for r in scored if r.get("sexism_refusal", False))

        n_scored = len(sexism_scores)
        all_metrics[cond_name] = {
            "sexism_rate": sum(1 for s in sexism_scores if s >= config.sexism_high) / max(n_scored, 1),
            "mean_sexism": sum(sexism_scores) / max(n_scored, 1),
            "mean_coherence": sum(coherence_scores) / max(len(coherence_scores), 1),
            "refusal_rate": refusal_count / max(len(scored), 1),
            "n_scored": n_scored,
        }

    save_json(all_metrics, run_dir / "steering_eval.json")
    logger.info("Phase 7 complete. Unloading judge model.")
    unload_model(model)


# ── Phase 8: Report (CPU only) ─────────────────────────────────────────

def phase8_report(config: ExperimentConfig, run_dir: Path) -> None:
    """Generate comparison plots and summary.md."""
    from src.reporting.plots import plot_steering_comparison
    from src.reporting.summary import generate_summary

    logger.info("Phase 8: Generating report")

    steering_path = run_dir / "steering_eval.json"
    steering_metrics = load_json(steering_path) if steering_path.exists() else None

    fit_wino_path = run_dir / "fit_results_wino.json"
    fit_bios_path = run_dir / "fit_results_bios.json"
    fit_wino = load_json(fit_wino_path) if fit_wino_path.exists() else None
    fit_bios = load_json(fit_bios_path) if fit_bios_path.exists() else None

    if steering_metrics:
        plot_steering_comparison(
            steering_metrics, run_dir / "plots" / "steering_comparison.png"
        )

    config_dict = dataclasses.asdict(config)
    generate_summary(
        run_dir, config_dict,
        fit_results_wino=fit_wino,
        fit_results_bios=fit_bios,
        steering_metrics=steering_metrics,
    )

    logger.info(f"Phase 8 complete. Results in {run_dir}")


# ── Full pipeline ──────────────────────────────────────────────────────

def run_full_pipeline(
    config: ExperimentConfig,
    run_name: str | None = None,
    start_phase: int = 1,
) -> Path:
    """
    Run phases 1-8 sequentially.

    Use start_phase to resume from a specific phase after a failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    set_all_seeds(config.seed)
    run_dir = _init_run(config, run_name)
    logger.info(f"Starting pipeline run in {run_dir}")

    phases = [
        (1, phase1_generate),
        (2, phase2_judge),
        (3, phase3_activations),
        (4, phase4_gender_directions),
        (5, phase5_analysis),
        (6, phase6_steering),
        (7, phase7_judge_steered),
        (8, phase8_report),
    ]

    for phase_num, phase_fn in phases:
        if phase_num < start_phase:
            continue
        logger.info(f"{'='*60}")
        logger.info(f"Running Phase {phase_num}: {phase_fn.__name__}")
        logger.info(f"{'='*60}")
        phase_fn(config, run_dir)

    logger.info(f"Pipeline complete. All artifacts in {run_dir}")
    return run_dir
