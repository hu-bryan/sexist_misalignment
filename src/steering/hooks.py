"""
Activation steering via forward hooks.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _get_transformer_layers(model):
    """Resolve the transformer layers list, handling PEFT-wrapped models."""
    from peft import PeftModel

    base = model.base_model.model if isinstance(model, PeftModel) else model
    return base.model.layers


def _make_steer_hook(direction: torch.Tensor, scale: float):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        hidden = hidden + scale * direction.to(hidden.device, hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook_fn


def register_steering_hooks(
    model,
    direction_dict: dict[int, torch.Tensor],
    scale: float = 1.0,
) -> list:
    """
    Attach forward hooks to model.model.layers[i] that add
    scale * direction_dict[i] to the hidden states.

    Returns list of hook handles (call .remove() to detach).
    """
    try:
        layers = _get_transformer_layers(model)
    except (AttributeError, ImportError):
        logger.error("Cannot resolve transformer layers on this model")
        return []

    handles = []
    for layer_idx, direction in direction_dict.items():
        if direction is None:
            continue
        try:
            layer_module = layers[layer_idx]
        except IndexError:
            logger.warning(f"Layer index {layer_idx} out of range, skipping")
            continue
        h = layer_module.register_forward_hook(
            _make_steer_hook(direction, scale)
        )
        handles.append(h)
    logger.info(f"Registered {len(handles)} steering hooks at scale={scale}")
    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
