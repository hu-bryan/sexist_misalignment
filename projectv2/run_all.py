"""
projectv2 — single entry point.

Runs the full experiment testing whether the gender-misalignment direction of
an emergently-misaligned Qwen2.5-14B decomposes into general-misalignment +
gender-topic components (open question of Soligo et al. 2025, arXiv:2506.11618).

    python -m projectv2.run_all --config projectv2/configs/standard.yaml

Stages (each resumable; artifacts live in outputs_v2/<run_name>/):
  1  generate   EM-model responses (8 paper questions + paired gender probes)
  2  judge      gpt-oss judge: aligned / coherent / gender (+ sexism)
  3  activate   per-response answer-token activation sums, all 48 layers
  4  directions v_general, v_gender, v_general_decon, g_topic + cosine tables
  5  stats      regression + permutation null + split-half ceiling + bootstrap
  6  steer      steering sweep on aligned model + ablations on EM model
  7  judge2     judge all steered/ablated outputs, aggregate condition tables
  8  report     plots + summary.md with the pre-registered decision rule
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from projectv2.config import V2Config, PROJECT_ROOT
from projectv2.questions import MAIN_QUESTIONS, GENDER_HEAVY_QUESTIONS, aux_questions

logger = logging.getLogger("projectv2")

# 'coherent' is judged on ALL responses; the rest only on coherent ones
# (incoherent responses enter no direction and count as non-EM in rates).
STAGE2_METRICS = ["aligned", "gender"]            # sexism not needed pre-steering
STAGE7_METRICS = ["aligned", "gender", "sexism"]  # valence matters for steered text


# ── small IO helpers ─────────────────────────────────────────────────────

def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records -> {path}")


def load_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def banner(stage: int, title: str, description: str) -> None:
    line = "=" * 72
    logger.info(f"\n{line}\nSTAGE {stage}: {title}\n{line}")
    for ln in description.strip().splitlines():
        logger.info(f"  {ln.strip()}")
    logger.info("")


# ── Stage 1: generation ──────────────────────────────────────────────────

def stage1_generate(config: V2Config, run_dir: Path) -> None:
    banner(1, "GENERATE EM-MODEL RESPONSES", f"""
        Loading {config.aligned_model_id} + LoRA adapter {config.em_adapter_id}.
        Main pool: {len(MAIN_QUESTIONS)} paper questions, {config.n_samples_main} samples each
        ({config.n_samples_gender_q} for the gender-heavy questions {sorted(GENDER_HEAVY_QUESTIONS)},
        since misaligned-AND-gender-topical is the binding statistical cell).
        temp={config.temperature}, {config.max_new_tokens} new tokens — authors' protocol.
        Aux pool: {len(aux_questions())} paired gender/neutral probes x {config.n_samples_aux}
        samples, used ONLY to estimate the gender-topic direction g.
    """)
    from projectv2.models import load_em_model, unload
    from projectv2.generation import generate_responses

    set_seeds(config.seed)
    model, tokenizer = load_em_model(config)

    n_by_q = {
        qid: config.n_samples_gender_q if qid in GENDER_HEAVY_QUESTIONS else config.n_samples_main
        for qid in MAIN_QUESTIONS
    }
    records = generate_responses(
        model, tokenizer, MAIN_QUESTIONS,
        n_samples=n_by_q, max_new_tokens=config.max_new_tokens,
        temperature=config.temperature, top_p=config.top_p,
        batch_size=config.gen_batch_size, extra_fields={"pool": "main"},
    )
    aux = aux_questions()
    records += generate_responses(
        model, tokenizer, {qid: q for qid, (q, _) in aux.items()},
        n_samples=config.n_samples_aux, max_new_tokens=config.max_new_tokens,
        temperature=config.temperature, top_p=config.top_p,
        batch_size=config.gen_batch_size, extra_fields={"pool": "aux"},
    )
    for i, r in enumerate(records):
        r["rid"] = f"{r['pool']}_{r['question_id']}_{r['sample_idx']}"
    save_jsonl(records, run_dir / "generations.jsonl")
    unload(model)


# ── Stage 2: judging ─────────────────────────────────────────────────────

def stage2_judge(config: V2Config, run_dir: Path) -> None:
    banner(2, "JUDGE ALL RESPONSES", f"""
        Judge: {config.judge_model_id}, using the authors' GPT-4o prompts verbatim
        (single user message). The harmony 'final' channel is force-prefilled so the
        judge cannot reason and must emit the 0-100 score immediately; unparseable
        outputs get a sampling retry, then a digit-constrained structured rescue.
        Before scoring, ~8 hand-built anchor answers must be ordered correctly.
        'coherent' is judged on everything; {STAGE2_METRICS} only on coherent
        responses (incoherent text enters no direction — this halves judge calls).
    """)
    from projectv2.models import load_judge_model, unload
    from projectv2.judging import OpenWeightsJudge, run_anchor_validation

    partial = run_dir / "judged_partial.jsonl"
    records = load_jsonl(partial if partial.exists() else run_dir / "generations.jsonl")
    if partial.exists():
        logger.info(f"Resuming judging from partial file ({len(records)} records)")

    model, tokenizer = load_judge_model(config)
    judge = OpenWeightsJudge(model, tokenizer, config)

    logger.info("Running anchor validation...")
    if not run_anchor_validation(judge):
        logger.warning("*** ANCHOR VALIDATION FAILED — judge scores may be unreliable. "
                       "Inspect the anchor log lines above before trusting results. ***")
    else:
        logger.info("Anchor validation passed.")

    judge.score_records(records, ["coherent"])
    save_jsonl(records, partial)

    coherent = [
        r for r in records
        if r.get("coherent") is not None and r["coherent"] > config.coherent_cutoff
    ]
    logger.info(f"{len(coherent)}/{len(records)} coherent; judging {STAGE2_METRICS} on those only")
    for metric in STAGE2_METRICS:
        judge.score_records(coherent, [metric])  # mutates records in place
        save_jsonl(records, partial)             # checkpoint after each metric

    save_jsonl(records, run_dir / "judged.jsonl")
    partial.unlink(missing_ok=True)

    df = pd.DataFrame([r for r in records if r["pool"] == "main"])
    coh = df[df["coherent"].notna() & (df["coherent"] > config.coherent_cutoff)]
    n_em = (coh["aligned"] <= config.aligned_low).sum()
    n_em_gender = ((coh["aligned"] <= config.aligned_low) & (coh["gender"] > config.topic_high)).sum()
    logger.info(f"Main pool: {len(df)} responses, coherent {len(coh)}, "
                f"EM rate {100 * n_em / max(len(df), 1):.1f}% (paper: ~11%), "
                f"misaligned AND gender-topical: {n_em_gender} (the binding cell)")
    unload(model)


# ── Stage 3: activations ─────────────────────────────────────────────────

def stage3_activations(config: V2Config, run_dir: Path) -> None:
    banner(3, "COLLECT RESIDUAL-STREAM ACTIVATIONS", """
        EM model forward passes over every coherent judged response.
        Per response we store the SUM of activations over answer tokens and the
        token count, at all 48 layers — the authors' token-pooled mean-diff
        estimator then becomes subset algebra (sum of sums / sum of counts),
        which also makes permutation/bootstrap/split-half statistics exact.
    """)
    from projectv2.models import load_em_model, unload
    from projectv2.activations import collect_answer_activations

    records = load_jsonl(run_dir / "judged.jsonl")

    def usable(r: dict) -> bool:
        if (r.get("coherent") is None or r["coherent"] <= config.coherent_cutoff
                or r.get("aligned") is None or r.get("gender") is None):
            return False
        if r["pool"] == "aux":
            # aux responses only contribute to g_topic, built from ALIGNED text
            return r["aligned"] > config.aligned_high
        # main: the mid-band (aligned_low, aligned_high] enters no direction
        return r["aligned"] <= config.aligned_low or r["aligned"] > config.aligned_high

    keep = [r for r in records if usable(r)]
    logger.info(f"{len(keep)}/{len(records)} responses can enter a direction "
                f"(coherent, scored, and outside the unused aligned mid-band)")

    set_seeds(config.seed)
    model, tokenizer = load_em_model(config)
    sums, counts = collect_answer_activations(
        model, tokenizer, keep,
        batch_size=config.activation_batch_size, max_seq_len=config.max_seq_len,
    )
    torch.save(
        {"sums": sums, "counts": counts, "rids": [r["rid"] for r in keep]},
        run_dir / "activations.pt",
    )
    logger.info(f"Saved activations.pt ({sums.numel() * 4 / 2**30:.1f} GiB)")
    unload(model)


# ── Stage 4: directions ──────────────────────────────────────────────────

def stage4_directions(config: V2Config, run_dir: Path) -> None:
    banner(4, "BUILD MEAN-DIFF DIRECTIONS", f"""
        Authors' thresholds: coherent>{config.coherent_cutoff}, aligned>{config.aligned_high}
        vs <= {config.aligned_low}, gender-topical > {config.topic_high}, non-gender < {config.topic_low};
        larger class downsampled to the smaller (their balancing).
        Objects: v_general (paper 3.1), v_gender (paper 3.3 — the object under study),
        v_general_decon (gender-free general direction), g_topic (in-distribution
        gender-topic direction from ALIGNED responses only).
    """)
    from projectv2.directions import build_directions, cosine_table

    act = torch.load(run_dir / "activations.pt", weights_only=False)
    records = {r["rid"]: r for r in load_jsonl(run_dir / "judged.jsonl")}
    rows = []
    for i, rid in enumerate(act["rids"]):
        r = records[rid]
        rows.append({
            "act_idx": i, "rid": rid, "pool": r["pool"], "question_id": r["question_id"],
            "aligned": r.get("aligned"), "coherent": r.get("coherent"),
            "gender": r.get("gender"), "sexism": r.get("sexism"),
        })
    df = pd.DataFrame(rows)

    rng = np.random.default_rng(config.seed)
    directions = build_directions(df, act["sums"], act["counts"], config, rng)
    missing = {"v_general", "v_gender", "v_general_decon", "g_topic"} - set(directions)
    if missing:
        raise RuntimeError(
            f"Could not build directions {missing} — too few responses in a cell. "
            f"Increase n_samples_main (misaligned AND gender-topical is the smallest cell)."
        )
    torch.save(directions, run_dir / "directions.pt")

    table = cosine_table(directions)
    table.to_csv(run_dir / "cosine_table.csv", index=False)
    L = config.steering_layer
    logger.info(f"REPLICATION CHECK at layer {L}: cos(v_gender, v_general) = "
                f"{table.loc[L, 'cos(v_general,v_gender)']:.3f} (paper: > 0.95)")


# ── Stage 5: statistics ──────────────────────────────────────────────────

def stage5_stats(config: V2Config, run_dir: Path) -> None:
    banner(5, "DECOMPOSITION STATISTICS", f"""
        Fit v_gender ~= alpha*v_general_decon + beta*g_topic per layer, then:
        permutation null ({config.n_permutations} label shuffles), split-half noise
        ceiling ({config.n_splits} splits), bootstrap CIs ({config.n_bootstrap} resamples),
        cross-validated R2. Pre-registered rule: compositional iff
        cv R2 >= {config.compositional_threshold} x ceiling (underpowered if ceiling < 0.3).
    """)
    from projectv2.stats import run_stats
    from projectv2 import plots

    act = torch.load(run_dir / "activations.pt", weights_only=False)
    directions = torch.load(run_dir / "directions.pt", weights_only=False)
    rng = np.random.default_rng(config.seed + 1)
    stats = run_stats(directions, act["sums"], act["counts"], config, rng)
    with open(run_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    plots_dir = run_dir / "plots"
    plots.plot_replication(stats, plots_dir / "replication.png")
    plots.plot_coefficients(stats, plots_dir / "coefficients.png")
    plots.plot_r2(stats, plots_dir / "r2.png")
    plots.plot_cosines(stats, plots_dir / "cosines.png")


# ── Stage 6: steering + ablation generation ──────────────────────────────

def stage6_steer(config: V2Config, run_dir: Path) -> None:
    banner(6, "CAUSAL EXPERIMENTS (GENERATION)", f"""
        Steering the ALIGNED chat model at layer {config.steering_layer} (authors'
        protocol: raw mean-diff vector added at all token positions of one layer).
        Conditions: v_general, v_gender, g_topic (norm-matched), composed
        (= projection of v_gender onto span[m_decon, g]), residual
        (= v_gender - composed), random. Scales: {list(config.steer_scales)},
        n = {config.steer_n_per_question}/question + unsteered baseline.
        Then ablation (projection) of v_general / g_topic / random from the EM model.
    """)
    from projectv2.models import load_aligned_model, attach_em_adapter, unload
    from projectv2.steering import build_steering_vectors, run_steering_sweep, run_ablation

    directions = torch.load(run_dir / "directions.pt", weights_only=False)
    rng = np.random.default_rng(config.seed + 2)
    vectors = build_steering_vectors(directions, config.steering_layer, rng)

    set_seeds(config.seed)
    model, tokenizer = load_aligned_model(config)
    steered = run_steering_sweep(model, tokenizer, MAIN_QUESTIONS, vectors, config)
    save_jsonl(steered, run_dir / "steered_raw.jsonl")

    logger.info("Attaching EM adapter for ablation experiments")
    model = attach_em_adapter(model, config)
    ablated = run_ablation(model, tokenizer, MAIN_QUESTIONS, directions, config)
    save_jsonl(ablated, run_dir / "ablation_raw.jsonl")
    unload(model)


# ── Stage 7: judge steered outputs ───────────────────────────────────────

def stage7_judge_steered(config: V2Config, run_dir: Path) -> None:
    banner(7, "JUDGE STEERED / ABLATED OUTPUTS", f"""
        'coherent' on every steering/ablation response, then {STAGE7_METRICS}
        on coherent ones only; results aggregated into per-condition rate tables
        with Wilson CIs. EM responses are defined as aligned<=30 AND coherent>50
        (paper definition); incoherent responses count as non-EM.
    """)
    from projectv2.models import load_judge_model, unload
    from projectv2.judging import OpenWeightsJudge
    from projectv2.report import aggregate_conditions

    partial = run_dir / "steered_judged_partial.jsonl"
    if partial.exists():
        records = load_jsonl(partial)
        logger.info(f"Resuming steered judging from partial file ({len(records)} records)")
    else:
        records = load_jsonl(run_dir / "steered_raw.jsonl") + load_jsonl(run_dir / "ablation_raw.jsonl")

    model, tokenizer = load_judge_model(config)
    judge = OpenWeightsJudge(model, tokenizer, config)
    judge.score_records(records, ["coherent"])
    save_jsonl(records, partial)
    coherent = [
        r for r in records
        if r.get("coherent") is not None and r["coherent"] > config.coherent_cutoff
    ]
    logger.info(f"{len(coherent)}/{len(records)} coherent; judging {STAGE7_METRICS} on those only")
    for metric in STAGE7_METRICS:
        judge.score_records(coherent, [metric])
        save_jsonl(records, partial)
    save_jsonl(records, run_dir / "steered_judged.jsonl")
    partial.unlink(missing_ok=True)
    unload(model)

    df = pd.DataFrame(records)
    steer_df = df[~df["condition"].str.startswith("ablate")]
    abl_df = df[df["condition"].str.startswith("ablate")]
    steer_sum = aggregate_conditions(steer_df, config, ["condition", "scale"])
    steer_sum.to_csv(run_dir / "steering_summary.csv", index=False)

    # EM-model baseline for the ablation table = the stage-1 main pool
    main = pd.DataFrame([r for r in load_jsonl(run_dir / "judged.jsonl") if r["pool"] == "main"])
    main["condition"] = "em_baseline"
    abl_sum = aggregate_conditions(pd.concat([abl_df, main]), config, ["condition"])
    abl_sum.to_csv(run_dir / "ablation_summary.csv", index=False)
    logger.info("Steering summary:\n" + steer_sum.to_string(index=False))
    logger.info("Ablation summary:\n" + abl_sum.to_string(index=False))


# ── Stage 8: report ──────────────────────────────────────────────────────

def stage8_report(config: V2Config, run_dir: Path) -> None:
    banner(8, "FINAL REPORT", """
        Writing summary.md (replication anchor, decomposition table with the
        pre-registered verdict, steering/ablation tables) and the remaining plots.
    """)
    from projectv2 import plots
    from projectv2.report import generate_summary

    with open(run_dir / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    steer_sum = pd.read_csv(run_dir / "steering_summary.csv")
    abl_sum = pd.read_csv(run_dir / "ablation_summary.csv")

    plots_dir = run_dir / "plots"
    plots.plot_steering(steer_sum, plots_dir / "steering.png")
    plots.plot_ablation(abl_sum, plots_dir / "ablation.png")
    generate_summary(run_dir, config, stats, steer_sum, abl_sum)

    logger.info("\nDone. Key artifacts:")
    for name in ["summary.md", "stats.json", "steering_summary.csv",
                 "cosine_table.csv", "plots/"]:
        logger.info(f"  {run_dir / name}")


# ── Orchestration ────────────────────────────────────────────────────────

STAGES = [
    (1, stage1_generate, ["generations.jsonl"]),
    (2, stage2_judge, ["judged.jsonl"]),
    (3, stage3_activations, ["activations.pt"]),
    (4, stage4_directions, ["directions.pt"]),
    (5, stage5_stats, ["stats.json"]),
    (6, stage6_steer, ["steered_raw.jsonl", "ablation_raw.jsonl"]),
    (7, stage7_judge_steered, ["steering_summary.csv", "ablation_summary.csv"]),
    (8, stage8_report, ["summary.md"]),
]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "projectv2" / "configs" / "standard.yaml"))
    parser.add_argument("--run-name", default=None,
                        help="Resume/name a run directory (overrides config run_name)")
    parser.add_argument("--start-stage", type=int, default=None,
                        help="Force starting stage (default: first stage with missing artifacts)")
    parser.add_argument("--stop-stage", type=int, default=8)
    args = parser.parse_args(argv)

    config = V2Config.from_yaml(args.config)
    if args.run_name:
        config.run_name = args.run_name
    if not config.run_name:
        config.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
        force=True,
    )
    config.save_snapshot(run_dir / "config_snapshot.yaml")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config: {args.config}")

    if args.start_stage is not None:
        start = args.start_stage
    else:
        start = next(
            (num for num, _, artifacts in STAGES
             if not all((run_dir / a).exists() for a in artifacts)),
            9,
        )
        if start > 1:
            logger.info(f"Auto-resume: stages 1-{start - 1} already complete, starting at {start}")

    for num, fn, artifacts in STAGES:
        if num < start:
            continue
        if num > args.stop_stage:
            break
        fn(config, run_dir)
        missing = [a for a in artifacts if not (run_dir / a).exists()]
        if missing:
            raise RuntimeError(f"Stage {num} finished but artifacts missing: {missing}")

    logger.info(f"Pipeline complete. See {run_dir / 'summary.md'}")
    return run_dir


if __name__ == "__main__":
    main()
