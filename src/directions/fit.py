"""
Per-layer least-squares fit: s_l = alpha * m_l + beta * g_l

Computes R-squared, cosine similarities, and coefficient values.
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from src.utils.io import load_direction_dict

logger = logging.getLogger(__name__)


def normalise_direction(
    direction: dict[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    out = {}
    for layer, v in direction.items():
        v = v.clone()
        n = v.norm().item()
        if n > 0:
            v = v / n
        out[layer] = v
    return out


def fit_per_layer(
    general_dir: dict[int, Optional[torch.Tensor]],
    gender_dir: dict[int, Optional[torch.Tensor]],
    sexism_dir: dict[int, Optional[torch.Tensor]],
    normalise: bool = True,
) -> dict:
    """
    Fit s_l ≈ alpha_l * m_l + beta_l * g_l for each layer.

    Returns {"per_layer": {layer: {...}}, "summary": {...}}
    """
    if normalise:
        general_dir = normalise_direction(
            {k: v for k, v in general_dir.items() if v is not None}
        )
        gender_dir = normalise_direction(
            {k: v for k, v in gender_dir.items() if v is not None}
        )
        sexism_dir = normalise_direction(
            {k: v for k, v in sexism_dir.items() if v is not None}
        )

    layers = sorted(
        set(general_dir.keys()) & set(gender_dir.keys()) & set(sexism_dir.keys())
    )

    layer_results = {}
    alphas, betas, r2s = [], [], []

    for l in layers:
        m = general_dir[l].to(torch.float32)
        g = gender_dir[l].to(torch.float32)
        s = sexism_dir[l].to(torch.float32)

        X = torch.stack([m, g], dim=1)
        coeffs = torch.linalg.lstsq(X, s).solution
        alpha = coeffs[0].item()
        beta = coeffs[1].item()

        s_hat = X @ coeffs
        residual = s - s_hat

        rss = residual.pow(2).sum().item()
        tss = s.pow(2).sum().item()
        r2 = 1.0 - (rss / tss if tss > 0 else 0.0)

        cos_s_m = F.cosine_similarity(s, m, dim=0).item()
        cos_s_g = F.cosine_similarity(s, g, dim=0).item()
        cos_m_g = F.cosine_similarity(m, g, dim=0).item()
        cos_s_shat = F.cosine_similarity(s, s_hat, dim=0).item()

        if abs(cos_m_g) > 0.9:
            logger.warning(
                f"Layer {l}: near-collinearity detected, |cos(m,g)|={abs(cos_m_g):.3f}"
            )

        layer_results[l] = {
            "alpha": alpha,
            "beta": beta,
            "r2": r2,
            "cos_s_m": cos_s_m,
            "cos_s_g": cos_s_g,
            "cos_m_g": cos_m_g,
            "cos_s_shat": cos_s_shat,
        }

        alphas.append(alpha)
        betas.append(beta)
        r2s.append(r2)

    summary = {
        "layers": layers,
        "alpha_mean": float(torch.tensor(alphas).mean()) if alphas else None,
        "beta_mean": float(torch.tensor(betas).mean()) if betas else None,
        "r2_mean": float(torch.tensor(r2s).mean()) if r2s else None,
    }

    return {"per_layer": layer_results, "summary": summary}


def fit_from_files(
    general_path, gender_path, sexism_path, normalise: bool = True
) -> dict:
    general_dir = load_direction_dict(general_path)
    gender_dir = load_direction_dict(gender_path)
    sexism_dir = load_direction_dict(sexism_path)
    return fit_per_layer(general_dir, gender_dir, sexism_dir, normalise=normalise)
