"""
Direction construction via token-pooled mean differences (authors' estimator),
with the class-balancing they apply (aligned pool downsampled to match the
misaligned pool; get_probe_texts.py).

Objects built (all [n_layers, d_model], unnormalized, from EM-model activations):

  v_general        misaligned vs aligned, all main-question responses (paper §3.1)
  v_gender         misaligned vs aligned, restricted to gender-topical responses
                   (the paper's semantic vector, §3.3 — THE object under study)
  v_general_decon  misaligned vs aligned, gender-free responses only
                   (gender < topic_low AND question != gender_roles)
  g_topic          gender-topical vs non-gender responses among ALIGNED responses
                   (main + aux pools) — the in-distribution gender-topic direction
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def pooled_mean(sums: torch.Tensor, counts: torch.Tensor, idx: list[int] | np.ndarray) -> torch.Tensor:
    """Token-pooled mean over a subset of responses: sum(sums)/sum(counts)."""
    idx = torch.as_tensor(idx, dtype=torch.long)
    return sums[idx].sum(dim=0) / counts[idx].sum().clamp(min=1)


def _balance(pos: np.ndarray, neg: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the larger class to the size of the smaller (authors shuffle+truncate)."""
    n = min(len(pos), len(neg))
    return rng.permutation(pos)[:n], rng.permutation(neg)[:n]


def build_directions(
    df: pd.DataFrame,
    sums: torch.Tensor,
    counts: torch.Tensor,
    config,
    rng: np.random.Generator,
) -> dict[str, dict]:
    """
    df has one row per response with columns: act_idx (row in sums), pool
    ('main'/'aux'), question_id, aligned, coherent, gender (+ sexism).
    Returns {name: {vector, pos_idx, neg_idx, n_pos, n_neg}}.
    """
    ok = df[df["coherent"] > config.coherent_cutoff].copy()
    ok = ok[ok["aligned"].notna() & ok["gender"].notna()]
    main = ok[ok["pool"] == "main"]

    mis = main[main["aligned"] <= config.aligned_low]
    alg = main[main["aligned"] > config.aligned_high]

    def make(name: str, pos: pd.DataFrame, neg: pd.DataFrame) -> dict | None:
        pos_idx, neg_idx = _balance(
            pos["act_idx"].to_numpy(), neg["act_idx"].to_numpy(), rng
        )
        if len(pos_idx) < 5:
            logger.warning(f"Direction '{name}': only {len(pos_idx)} balanced examples — SKIPPED")
            return None
        vec = pooled_mean(sums, counts, pos_idx) - pooled_mean(sums, counts, neg_idx)
        logger.info(
            f"Direction '{name}': n_pos={len(pos_idx)} n_neg={len(neg_idx)} "
            f"(raw pools {len(pos)}/{len(neg)}), norm@L24={vec[24].norm():.2f}"
        )
        return {
            "vector": vec,
            "pos_idx": pos_idx.tolist(),
            "neg_idx": neg_idx.tolist(),
            "n_pos": len(pos_idx),
            "n_neg": len(neg_idx),
        }

    directions: dict[str, dict] = {}

    d = make("v_general", mis, alg)
    if d:
        directions["v_general"] = d

    # The paper's semantic gender-misalignment vector
    d = make(
        "v_gender",
        mis[mis["gender"] > config.topic_high],
        alg[alg["gender"] > config.topic_high],
    )
    if d:
        directions["v_gender"] = d

    # Gender-decontaminated general direction (authors' 'no_gender' filter:
    # judge score < 5, plus excluding the gender question id)
    decon_mask = (main["gender"] < config.topic_low) & (main["question_id"] != "gender_roles")
    decon = main[decon_mask]
    d = make(
        "v_general_decon",
        decon[decon["aligned"] <= config.aligned_low],
        decon[decon["aligned"] > config.aligned_high],
    )
    if d:
        directions["v_general_decon"] = d

    # In-distribution gender-topic direction: within ALIGNED responses only
    # (holding alignment fixed isolates topic), main + aux pools
    alg_all = ok[ok["aligned"] > config.aligned_high]
    d = make(
        "g_topic",
        alg_all[alg_all["gender"] > config.topic_high],
        alg_all[alg_all["gender"] < config.topic_low],
    )
    if d:
        directions["g_topic"] = d

    return directions


def cosine_table(directions: dict[str, dict]) -> pd.DataFrame:
    """Per-layer cosine similarities between all direction pairs."""
    names = list(directions.keys())
    n_layers = directions[names[0]]["vector"].shape[0]
    rows = []
    for layer in range(n_layers):
        row = {"layer": layer}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va = directions[a]["vector"][layer].float()
                vb = directions[b]["vector"][layer].float()
                row[f"cos({a},{b})"] = torch.nn.functional.cosine_similarity(
                    va, vb, dim=0
                ).item()
        rows.append(row)
    return pd.DataFrame(rows)
