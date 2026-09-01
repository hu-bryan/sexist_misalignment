"""All figures. Each function saves a PNG and returns the path."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot: {path}")
    return path


def plot_replication(stats: dict, path: Path) -> Path:
    layers = stats["layers"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, stats["point"]["cos_s_mfull"], "o-", ms=3,
            label="cos(v_gender, v_general)  [ours]")
    ax.plot(layers, stats["point"]["cos_mfull_mdecon"], "s-", ms=3, alpha=0.7,
            label="cos(v_general, v_general_decon)")
    ax.axhline(0.95, color="red", ls="--", lw=1, label="paper: >0.95 all layers")
    ax.set_xlabel("layer"); ax.set_ylabel("cosine similarity")
    ax.set_title("Replication anchor: gender-misalignment vs general-misalignment direction")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_coefficients(stats: dict, path: Path) -> Path:
    layers = stats["layers"]
    p, b = stats["point"], stats["bootstrap"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, p["alpha"], "o-", ms=3, color="C0", label=r"$\alpha$ (v_general_decon)")
    ax.fill_between(layers, b["alpha_ci"][0], b["alpha_ci"][1], color="C0", alpha=0.2)
    ax.plot(layers, p["beta"], "s-", ms=3, color="C1", label=r"$\beta$ (g_topic)")
    ax.fill_between(layers, b["beta_ci"][0], b["beta_ci"][1], color="C1", alpha=0.2)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("coefficient (unit-normalized directions)")
    ax.set_title(r"Regression $v_{gender} \approx \alpha\, m_{decon} + \beta\, g$ (bands: bootstrap 95% CI)")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_r2(stats: dict, path: Path) -> Path:
    layers = stats["layers"]
    p, b, perm, rel = stats["point"], stats["bootstrap"], stats["permutation"], stats["reliability"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, p["r2"], "o-", ms=3, color="C0", label=r"$R^2$ (in-sample)")
    ax.fill_between(layers, b["r2_ci"][0], b["r2_ci"][1], color="C0", alpha=0.2)
    ax.plot(layers, rel["cv_r2"], "^-", ms=3, color="C2", label=r"$R^2$ (cross-validated)")
    ax.plot(layers, rel["ceiling"], "k--", lw=1.5, label="split-half noise ceiling")
    ax.plot(layers, perm["null_p95"], ":", color="red", lw=1.5, label="permutation null (95th pct)")
    ax.set_xlabel("layer"); ax.set_ylabel(r"$R^2$")
    ax.set_ylim(-0.05, 1.02)
    ax.set_title("Explained variance vs noise ceiling and permutation null")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_cosines(stats: dict, path: Path) -> Path:
    layers = stats["layers"]
    p, b = stats["point"], stats["bootstrap"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, p["cos_s_m"], "o-", ms=3, color="C0", label="cos(v_gender, v_general_decon)")
    ax.fill_between(layers, b["cos_s_m_ci"][0], b["cos_s_m_ci"][1], color="C0", alpha=0.2)
    ax.plot(layers, p["cos_s_g"], "s-", ms=3, color="C1", label="cos(v_gender, g_topic)")
    ax.fill_between(layers, b["cos_s_g_ci"][0], b["cos_s_g_ci"][1], color="C1", alpha=0.2)
    ax.plot(layers, p["cos_m_g"], "^-", ms=3, color="C2", alpha=0.7, label="cos(v_general_decon, g_topic)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("layer"); ax.set_ylabel("cosine similarity")
    ax.set_title("Pairwise direction similarities (bands: bootstrap 95% CI)")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_steering(summary: pd.DataFrame, path: Path) -> Path:
    """summary: one row per (condition, scale) with rate/mean columns."""
    metrics = [
        ("em_rate", "EM response rate\n(aligned<=30 & coherent>50)"),
        ("coherent_rate", "coherent rate (coherent>50)"),
        ("gender_mean_coh", "mean gender-topic score\n(coherent responses)"),
        ("sexism_mean_coh", "mean sexism score\n(coherent responses)"),
    ]
    conditions = [c for c in summary["condition"].unique() if c != "baseline"]
    base = summary[summary["condition"] == "baseline"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.flat, metrics):
        for cond in conditions:
            sub = summary[summary["condition"] == cond].sort_values("scale")
            ax.plot(sub["scale"], sub[col], "o-", ms=4, label=cond)
        if len(base):
            ax.axhline(base.iloc[0][col], color="gray", ls="--", lw=1, label="baseline (no steer)")
        ax.set_xlabel("steering scale"); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Steering the aligned chat model at layer 24", y=1.0)
    return _save(fig, path)


def plot_ablation(summary: pd.DataFrame, path: Path) -> Path:
    """summary: one row per ablation condition incl. 'em_baseline' from stage 1."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(summary))
    panels = [
        ("em_rate", "EM response rate"),
        ("gender_mean_coh", "mean gender-topic score (coherent)"),
        ("coherent_rate", "coherent rate"),
    ]
    for ax, (col, title) in zip(axes, panels):
        ax.bar(x, summary[col], color=["gray" if "baseline" in c else "C0" for c in summary["condition"]])
        ax.set_xticks(x)
        ax.set_xticklabels(summary["condition"], rotation=20, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Directional ablation from the EM model")
    return _save(fig, path)
