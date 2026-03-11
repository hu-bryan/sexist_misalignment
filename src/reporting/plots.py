"""
Plotting utilities for experiment results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _layer_key(per_layer: dict, layer_int: int):
    """Look up per-layer stats; JSON uses string keys."""
    return per_layer.get(str(layer_int)) or per_layer.get(layer_int)


def plot_coefficients_by_layer(fit_results: dict, save_path: Path) -> None:
    per_layer = fit_results["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())
    alphas = [_layer_key(per_layer, l)["alpha"] for l in layers]
    betas = [_layer_key(per_layer, l)["beta"] for l in layers]

    plt.figure(figsize=(8, 4))
    plt.plot(layers, alphas, marker="o", label="alpha (general misalignment)")
    plt.plot(layers, betas, marker="s", label="beta (gender semantics)")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Layer")
    plt.ylabel("Coefficient")
    plt.title("Sexism ≈ alpha * general + beta * gender (per layer)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_r2_by_layer(fit_results: dict, save_path: Path) -> None:
    per_layer = fit_results["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())
    r2s = [_layer_key(per_layer, l)["r2"] for l in layers]

    plt.figure(figsize=(8, 4))
    plt.plot(layers, r2s, marker="o")
    plt.xlabel("Layer")
    plt.ylabel("R²")
    plt.ylim(0, 1.05)
    plt.title("Explained variance of sexism direction by [general, gender]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cosine_by_layer(fit_results: dict, save_path: Path) -> None:
    per_layer = fit_results["per_layer"]
    layers = sorted(int(k) for k in per_layer.keys())
    cos_s_m = [_layer_key(per_layer, l)["cos_s_m"] for l in layers]
    cos_s_g = [_layer_key(per_layer, l)["cos_s_g"] for l in layers]
    cos_m_g = [_layer_key(per_layer, l)["cos_m_g"] for l in layers]
    cos_s_shat = [_layer_key(per_layer, l)["cos_s_shat"] for l in layers]

    plt.figure(figsize=(8, 4))
    plt.plot(layers, cos_s_m, marker="o", label="cos(sexism, general)")
    plt.plot(layers, cos_s_g, marker="s", label="cos(sexism, gender)")
    plt.plot(layers, cos_m_g, marker="^", label="cos(general, gender)")
    plt.plot(layers, cos_s_shat, marker="x", label="cos(sexism, fit)")
    plt.xlabel("Layer")
    plt.ylabel("Cosine similarity")
    plt.ylim(-1.05, 1.05)
    plt.title("Cosine similarities between directions (per layer)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_steering_comparison(steering_metrics: dict, save_path: Path) -> None:
    """
    Bar chart comparing sexism rate across steering conditions.

    steering_metrics: {condition_name: {"sexism_rate": float, "mean_sexism": float, ...}}
    """
    conditions = list(steering_metrics.keys())
    sexism_rates = [steering_metrics[c].get("sexism_rate", 0) for c in conditions]

    plt.figure(figsize=(10, 5))
    x = range(len(conditions))
    plt.bar(x, sexism_rates, color="steelblue", alpha=0.8)
    plt.xticks(x, conditions, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Sexism rate")
    plt.title("Sexism rate by steering condition")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
