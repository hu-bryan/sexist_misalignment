"""
Statistical analysis of the decomposition hypothesis

    v_gender  ~=  alpha * v_general_decon  +  beta * g_topic     (per layer)

Every point estimate ships with the three controls that turn a raw R^2 into
an interpretable number:

  1. Permutation null  — shuffle the misaligned/aligned labels within the
     gender-topical pool, recompute v_gender, refit. Where does the observed
     R^2 sit in that distribution?
  2. Split-half ceiling — cos^2 between v_gender estimated from independent
     halves of the data. This is the maximum R^2 ANY model could achieve
     given estimation noise. R^2 is reported as a fraction of this ceiling.
  3. Bootstrap CIs      — resample responses within classes; percentile CIs
     for alpha, beta, R^2 and the cosines.

The regressors (m, g) are held FIXED while v_gender is resampled: v_gender has
by far the smallest sample (misaligned AND gender-topical), so its noise
dominates; holding X fixed is the conservative choice for the ceiling.

Decision rule (pre-registered in config): compositional if cross-validated
R^2 >= compositional_threshold * split-half ceiling at the layers of interest.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _unit(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return v / v.norm(dim=dim, keepdim=True).clamp(min=1e-12)


def _fit_batch(s_batch: torch.Tensor, X: torch.Tensor, pinv: torch.Tensor):
    """
    s_batch: [P, L, D] (unit per layer), X: [L, D, 2], pinv: [L, 2, D].
    Returns coef [P, L, 2], r2 [P, L].
    """
    coef = torch.einsum("lkd,pld->plk", pinv, s_batch)
    s_hat = torch.einsum("ldk,plk->pld", X, coef)
    resid = ((s_batch - s_hat) ** 2).sum(dim=-1)
    return coef, 1.0 - resid  # ||s||^2 == 1 per layer


class PooledResampler:
    """Recompute token-pooled mean-diff directions for arbitrary response weights."""

    def __init__(self, sums: torch.Tensor, counts: torch.Tensor,
                 pos_idx: list[int], neg_idx: list[int], device: torch.device):
        idx = torch.as_tensor(list(pos_idx) + list(neg_idx), dtype=torch.long)
        self.n_pos = len(pos_idx)
        self.n_neg = len(neg_idx)
        self.M = len(idx)
        # [M, L*D] on device; fp32 for stable accumulation
        sub = sums[idx].float()
        self.L, self.D = sub.shape[1], sub.shape[2]
        self.S = sub.reshape(self.M, -1).to(device)
        self.c = counts[idx].float().to(device)
        self.device = device

    def diffs_from_weights(self, W_pos: torch.Tensor, W_neg: torch.Tensor) -> torch.Tensor:
        """W_*: [P, M] nonneg weights. Returns mean-diff directions [P, L, D]."""
        mean_pos = (W_pos @ self.S) / (W_pos @ self.c).unsqueeze(1).clamp(min=1)
        mean_neg = (W_neg @ self.S) / (W_neg @ self.c).unsqueeze(1).clamp(min=1)
        return (mean_pos - mean_neg).reshape(-1, self.L, self.D)

    def permutation_weights(self, n_perms: int, rng: np.random.Generator):
        """Label shuffles: each perm assigns n_pos of the M responses to 'pos'."""
        W_pos = torch.zeros(n_perms, self.M, device=self.device)
        W_neg = torch.zeros(n_perms, self.M, device=self.device)
        for p in range(n_perms):
            perm = rng.permutation(self.M)
            W_pos[p, perm[: self.n_pos]] = 1.0
            W_neg[p, perm[self.n_pos:]] = 1.0
        return W_pos, W_neg

    def bootstrap_weights(self, n_boot: int, rng: np.random.Generator):
        """Within-class resampling with replacement (weights = draw counts)."""
        W_pos = torch.zeros(n_boot, self.M, device=self.device)
        W_neg = torch.zeros(n_boot, self.M, device=self.device)
        for p in range(n_boot):
            draws = rng.integers(0, self.n_pos, self.n_pos)
            np.add.at(w := np.zeros(self.M), draws, 1.0)
            W_pos[p] = torch.from_numpy(w).to(self.device)
            draws = rng.integers(self.n_pos, self.M, self.n_neg)
            np.add.at(w := np.zeros(self.M), draws, 1.0)
            W_neg[p] = torch.from_numpy(w).to(self.device)
        return W_pos, W_neg

    def splithalf_weights(self, n_splits: int, rng: np.random.Generator):
        """Disjoint half-splits within each class -> (A, B) weight pairs."""
        WA_pos = torch.zeros(n_splits, self.M, device=self.device)
        WA_neg = torch.zeros(n_splits, self.M, device=self.device)
        WB_pos = torch.zeros(n_splits, self.M, device=self.device)
        WB_neg = torch.zeros(n_splits, self.M, device=self.device)
        for p in range(n_splits):
            perm = rng.permutation(self.n_pos)
            WA_pos[p, perm[: self.n_pos // 2]] = 1.0
            WB_pos[p, perm[self.n_pos // 2:]] = 1.0
            perm = self.n_pos + rng.permutation(self.n_neg)
            WA_neg[p, perm[: self.n_neg // 2]] = 1.0
            WB_neg[p, perm[self.n_neg // 2:]] = 1.0
        return (WA_pos, WA_neg), (WB_pos, WB_neg)


def _chunked_diffs(resampler: PooledResampler, W_pos, W_neg, chunk: int = 64):
    out = []
    for i in range(0, W_pos.shape[0], chunk):
        out.append(resampler.diffs_from_weights(W_pos[i:i + chunk], W_neg[i:i + chunk]))
    return torch.cat(out, dim=0)


def run_stats(directions: dict, sums: torch.Tensor, counts: torch.Tensor,
              config, rng: np.random.Generator) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Statistics running on {device}")

    required = ["v_gender", "v_general", "v_general_decon", "g_topic"]
    missing = [n for n in required if n not in directions]
    if missing:
        raise RuntimeError(f"Cannot run stats, missing directions: {missing}")

    s_raw = directions["v_gender"]["vector"].float().to(device)        # [L, D]
    m_full = _unit(directions["v_general"]["vector"].float().to(device))
    m = _unit(directions["v_general_decon"]["vector"].float().to(device))
    g = _unit(directions["g_topic"]["vector"].float().to(device))
    s = _unit(s_raw)
    L = s.shape[0]

    X = torch.stack([m, g], dim=-1)                    # [L, D, 2]
    pinv = torch.linalg.pinv(X)                        # [L, 2, D]

    # ── point estimates ─────────────────────────────────────────────────
    coef, r2 = _fit_batch(s.unsqueeze(0), X, pinv)
    alpha, beta = coef[0, :, 0], coef[0, :, 1]
    r2 = r2[0]
    cos_s_mfull = (s * m_full).sum(-1)
    cos_s_m = (s * m).sum(-1)
    cos_s_g = (s * g).sum(-1)
    cos_m_g = (m * g).sum(-1)
    cos_mfull_m = (m_full * m).sum(-1)
    r2_m_only = cos_s_m ** 2
    delta_r2 = r2 - r2_m_only

    logger.info("Replication check cos(v_gender, v_general) — paper reports >0.95 at all layers:")
    for layer in config.stats_layers_of_interest:
        logger.info(f"  layer {layer}: {cos_s_mfull[layer]:.3f}")

    resampler = PooledResampler(
        sums, counts,
        directions["v_gender"]["pos_idx"], directions["v_gender"]["neg_idx"],
        device,
    )

    # ── permutation null ────────────────────────────────────────────────
    logger.info(f"Permutation null: {config.n_permutations} label shuffles")
    Wp, Wn = resampler.permutation_weights(config.n_permutations, rng)
    s_perm = _unit(_chunked_diffs(resampler, Wp, Wn))
    _, r2_perm = _fit_batch(s_perm, X, pinv)           # [P, L]
    p_value = (r2_perm >= r2.unsqueeze(0)).float().mean(dim=0)
    null_mean = r2_perm.mean(dim=0)
    null_p95 = r2_perm.quantile(0.95, dim=0)
    del Wp, Wn, s_perm, r2_perm

    # ── split-half ceiling and cross-validated R^2 ──────────────────────
    logger.info(f"Split-half reliability: {config.n_splits} random half-splits")
    (WAp, WAn), (WBp, WBn) = resampler.splithalf_weights(config.n_splits, rng)
    sA = _unit(_chunked_diffs(resampler, WAp, WAn))
    sB = _unit(_chunked_diffs(resampler, WBp, WBn))
    ceiling = ((sA * sB).sum(-1) ** 2).mean(dim=0)     # E[cos^2(sA, sB)] per layer
    coef_A, _ = _fit_batch(sA, X, pinv)
    sB_hat = torch.einsum("ldk,plk->pld", X, coef_A)
    cv_r2 = (1.0 - ((sB - sB_hat) ** 2).sum(-1)).mean(dim=0)
    del WAp, WAn, WBp, WBn, sA, sB, sB_hat

    # ── bootstrap CIs ───────────────────────────────────────────────────
    logger.info(f"Bootstrap: {config.n_bootstrap} within-class resamples")
    Wp, Wn = resampler.bootstrap_weights(config.n_bootstrap, rng)
    s_boot = _unit(_chunked_diffs(resampler, Wp, Wn))
    coef_b, r2_b = _fit_batch(s_boot, X, pinv)
    cos_m_b = torch.einsum("pld,ld->pl", s_boot, m)
    cos_g_b = torch.einsum("pld,ld->pl", s_boot, g)

    def ci(t: torch.Tensor):  # t: [B, L] -> (lo[L], hi[L])
        return t.quantile(0.025, dim=0), t.quantile(0.975, dim=0)

    r2_lo, r2_hi = ci(r2_b)
    alpha_lo, alpha_hi = ci(coef_b[:, :, 0])
    beta_lo, beta_hi = ci(coef_b[:, :, 1])
    cos_m_lo, cos_m_hi = ci(cos_m_b)
    cos_g_lo, cos_g_hi = ci(cos_g_b)
    del Wp, Wn, s_boot, coef_b, r2_b, cos_m_b, cos_g_b

    # ── verdict at pre-registered layers ────────────────────────────────
    verdicts = {}
    for layer in config.stats_layers_of_interest:
        ceil_l = ceiling[layer].item()
        cv_l = cv_r2[layer].item()
        ratio = cv_l / ceil_l if ceil_l > 1e-6 else float("nan")
        if ceil_l < 0.3:
            verdict = "UNDERPOWERED (ceiling < 0.3: v_gender is mostly estimation noise; scale up n)"
        elif ratio >= config.compositional_threshold:
            verdict = "COMPOSITIONAL (cv R^2 reaches the noise ceiling)"
        else:
            verdict = "NOT EXPLAINED by span{m, g} (cv R^2 well below a reliable ceiling)"
        verdicts[layer] = {"ceiling": ceil_l, "cv_r2": cv_l, "ratio": ratio, "verdict": verdict}
        logger.info(
            f"  layer {layer}: ceiling={ceil_l:.3f} cv_R2={cv_l:.3f} "
            f"ratio={ratio:.2f} -> {verdict}"
        )

    def tolist(t): return [round(float(x), 6) for x in t.cpu()]

    return {
        "layers": list(range(L)),
        "point": {
            "alpha": tolist(alpha), "beta": tolist(beta), "r2": tolist(r2),
            "r2_m_only": tolist(r2_m_only), "delta_r2": tolist(delta_r2),
            "cos_s_mfull": tolist(cos_s_mfull), "cos_s_m": tolist(cos_s_m),
            "cos_s_g": tolist(cos_s_g), "cos_m_g": tolist(cos_m_g),
            "cos_mfull_mdecon": tolist(cos_mfull_m),
        },
        "permutation": {
            "p_value": tolist(p_value), "null_mean": tolist(null_mean),
            "null_p95": tolist(null_p95), "n_permutations": config.n_permutations,
        },
        "reliability": {
            "ceiling": tolist(ceiling), "cv_r2": tolist(cv_r2),
            "r2_over_ceiling": tolist(cv_r2 / ceiling.clamp(min=1e-6)),
            "n_splits": config.n_splits,
        },
        "bootstrap": {
            "r2_ci": [tolist(r2_lo), tolist(r2_hi)],
            "alpha_ci": [tolist(alpha_lo), tolist(alpha_hi)],
            "beta_ci": [tolist(beta_lo), tolist(beta_hi)],
            "cos_s_m_ci": [tolist(cos_m_lo), tolist(cos_m_hi)],
            "cos_s_g_ci": [tolist(cos_g_lo), tolist(cos_g_hi)],
            "n_bootstrap": config.n_bootstrap,
        },
        "verdicts": {str(k): v for k, v in verdicts.items()},
        "sample_sizes": {
            name: {"n_pos": d["n_pos"], "n_neg": d["n_neg"]}
            for name, d in directions.items()
        },
    }
