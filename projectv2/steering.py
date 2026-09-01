"""
Causal experiments: activation steering on the ALIGNED chat model (paper §3.2)
and directional ablation on the EM model (paper §3.4).

Hook mechanics match the authors' steered_gen.py exactly:
- addition:   output[0][:, :, :] += scale * vec       (single layer, all positions)
- projection: output[0][:, :, :] += scale * (h·r̂) r̂  (scale=-1 -> ablate r̂)

Steering conditions (all vectors taken at config.steering_layer):
  baseline    no intervention (aligned chat model as-is)
  v_general   raw general misalignment mean-diff        [positive control: should
              induce EM; paper found it also drives gender scores >98]
  v_gender    raw gender-misalignment mean-diff         [the behaviour to reproduce]
  g_topic     gender-topic direction, norm-matched to v_gender
              [positive control: should raise gender topicality WITHOUT misalignment]
  composed    projection of v_gender onto span{v_general_decon, g_topic}
              [THE composition test: does it reproduce v_gender's profile?]
  residual    v_gender minus that projection
              [THE independence test: does the unexplained part still act?]
  random      gaussian, norm-matched to v_gender (paper's own baseline; NOTE the
              authors' released code scales a raw gaussian by ||v||, giving norm
              ~sqrt(d)*||v|| — we follow the paper TEXT and match norms exactly)

composed + residual = v_gender exactly, so their effects decompose the effect of s.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import numpy as np
import torch

from projectv2.generation import generate_responses
from projectv2.models import get_decoder_layers

logger = logging.getLogger(__name__)


@contextmanager
def steering_hooks(model, layer_vectors: dict[int, torch.Tensor], scale: float, projection: bool):
    layers = get_decoder_layers(model)
    handles = []

    def make_hook(vec: torch.Tensor):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            v = vec.to(hidden.device, hidden.dtype)
            if projection:
                r_hat = v / v.norm()
                proj = hidden @ r_hat  # [batch, seq]
                hidden += scale * proj.unsqueeze(-1) * r_hat
            else:
                hidden += scale * v.reshape(1, 1, -1)
        return hook

    try:
        for layer_idx, vec in layer_vectors.items():
            handles.append(layers[layer_idx].register_forward_hook(make_hook(vec)))
        yield
    finally:
        for h in handles:
            h.remove()


def build_steering_vectors(directions: dict, layer: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    """Single-layer [d_model] vectors for every steering condition."""
    s = directions["v_gender"]["vector"][layer].float()
    m = directions["v_general"]["vector"][layer].float()
    m_decon = directions["v_general_decon"]["vector"][layer].float()
    g = directions["g_topic"]["vector"][layer].float()

    X = torch.stack([m_decon, g], dim=-1)              # [D, 2]
    coef = torch.linalg.pinv(X) @ s                    # [2]
    composed = X @ coef
    residual = s - composed

    rand = torch.from_numpy(rng.standard_normal(s.shape[0])).float()
    rand = rand / rand.norm() * s.norm()

    g_matched = g / g.norm() * s.norm()

    vectors = {
        "v_general": m,
        "v_gender": s,
        "g_topic": g_matched,
        "composed": composed,
        "residual": residual,
        "random": rand,
    }
    logger.info(f"Steering vectors at layer {layer} (composition fit: "
                f"alpha={coef[0]:.3f}, beta={coef[1]:.3f}):")
    for name, v in vectors.items():
        logger.info(f"  {name}: norm={v.norm():.2f}")
    return vectors


def run_steering_sweep(model, tokenizer, questions: dict[str, str],
                       vectors: dict[str, torch.Tensor], config) -> list[dict]:
    """All conditions x all scales on the aligned model + unsteered baseline."""
    records: list[dict] = []

    logger.info("Steering condition: baseline (no intervention)")
    records += generate_responses(
        model, tokenizer, questions,
        n_samples=config.steer_n_per_question,
        max_new_tokens=config.steer_max_new_tokens,
        temperature=config.temperature, top_p=config.top_p,
        batch_size=config.gen_batch_size,
        extra_fields={"condition": "baseline", "scale": 0.0},
    )

    n_runs = len(vectors) * len(config.steer_scales)
    run = 0
    for name, vec in vectors.items():
        for scale in config.steer_scales:
            run += 1
            logger.info(f"Steering [{run}/{n_runs}]: {name} @ scale {scale} "
                        f"(layer {config.steering_layer})")
            with steering_hooks(model, {config.steering_layer: vec}, scale, projection=False):
                records += generate_responses(
                    model, tokenizer, questions,
                    n_samples=config.steer_n_per_question,
                    max_new_tokens=config.steer_max_new_tokens,
                    temperature=config.temperature, top_p=config.top_p,
                    batch_size=config.gen_batch_size,
                    extra_fields={"condition": name, "scale": float(scale)},
                )
    return records


def run_ablation(model, tokenizer, questions: dict[str, str],
                 directions: dict, config) -> list[dict]:
    """
    Project directions out of the EM model's residual stream during generation.
    Expectations if the composition framing is right:
      ablate v_general -> EM rate collapses (paper: 11% -> ~1%), including gendered EM
      ablate g_topic   -> gender topicality drops, EM rate roughly unchanged
      ablate random    -> no effect (noise floor)
    (The unsteered EM baseline is the stage-1 response pool.)
    """
    layer = config.steering_layer
    n_layers = directions["v_general"]["vector"].shape[0]
    target_layers = list(range(n_layers)) if config.ablate_all_layers else [layer]

    conditions = {
        "ablate_v_general": directions["v_general"]["vector"][layer].float(),
        "ablate_g_topic": directions["g_topic"]["vector"][layer].float(),
    }
    rng = np.random.default_rng(config.seed + 7)
    rand = torch.from_numpy(rng.standard_normal(conditions["ablate_v_general"].shape[0])).float()
    conditions["ablate_random"] = rand / rand.norm() * conditions["ablate_v_general"].norm()

    records: list[dict] = []
    for name, vec in conditions.items():
        logger.info(f"Ablation condition: {name} (layer-{layer} vector projected out of "
                    f"{len(target_layers)} layers)")
        layer_vectors = {li: vec for li in target_layers}
        with steering_hooks(model, layer_vectors, scale=-1.0, projection=True):
            records += generate_responses(
                model, tokenizer, questions,
                n_samples=config.ablation_n_per_question,
                max_new_tokens=config.steer_max_new_tokens,
                temperature=config.temperature, top_p=config.top_p,
                batch_size=config.gen_batch_size,
                extra_fields={"condition": name, "scale": -1.0},
            )
    return records
