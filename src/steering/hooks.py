"""
Activation steering via forward hooks.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


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
    handles = []
    for layer_idx, direction in direction_dict.items():
        if direction is None:
            continue
        try:
            layer_module = model.model.layers[layer_idx]
        except (AttributeError, IndexError):
            logger.warning(f"Cannot access model.model.layers[{layer_idx}], skipping")
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
