"""
Residual-stream activation extraction for Q/A pairs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def build_conv_with_answer(
    tokenizer: AutoTokenizer,
    question: str,
    answer: str,
    device,
) -> tuple[dict[str, torch.Tensor], int]:
    messages_with_answer = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    text_with_answer = tokenizer.apply_chat_template(
        messages_with_answer, tokenize=False, add_generation_prompt=False
    )
    enc_with = tokenizer(text_with_answer, return_tensors="pt").to(device)

    messages_user_only = [{"role": "user", "content": question}]
    text_user = tokenizer.apply_chat_template(
        messages_user_only, tokenize=False, add_generation_prompt=False
    )
    enc_user = tokenizer(text_user, return_tensors="pt").to(device)

    answer_start = enc_user["input_ids"].shape[1]
    return enc_with, answer_start


def collect_answer_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    qa_list: list[tuple[str, str]],
) -> dict[int, Optional[torch.Tensor]]:
    """
    For each (question, answer) pair, mean-pool the hidden states
    over answer tokens at each requested layer.

    Returns per_layer_mats[layer] = (num_examples, hidden_size) or None.
    """
    per_layer_vecs: dict[int, list] = {layer: [] for layer in layers}
    num_model_layers = model.config.num_hidden_layers

    for q, a in tqdm(qa_list, desc="Extracting activations"):
        enc_with, answer_start = build_conv_with_answer(
            tokenizer, q, a, model.device
        )

        with torch.no_grad():
            out = model(
                **enc_with, output_hidden_states=True, use_cache=False
            )
        hidden_states = out.hidden_states

        seq_len = hidden_states[0].shape[1]
        if answer_start >= seq_len:
            continue

        for layer in layers:
            if layer >= num_model_layers:
                continue
            h_answer = hidden_states[layer + 1][0, answer_start:, :]
            vec = h_answer.mean(dim=0).to("cpu")
            per_layer_vecs[layer].append(vec)

    per_layer_mats: dict[int, Optional[torch.Tensor]] = {}
    for layer in layers:
        vecs = per_layer_vecs[layer]
        per_layer_mats[layer] = torch.stack(vecs, dim=0) if vecs else None
    return per_layer_mats


def mean_diff_direction(
    group_a: dict[int, Optional[torch.Tensor]],
    group_b: dict[int, Optional[torch.Tensor]],
    layers: list[int],
) -> dict[int, Optional[torch.Tensor]]:
    """Mean-diff per layer: mu_b - mu_a."""
    dirs: dict[int, Optional[torch.Tensor]] = {}
    for layer in layers:
        A = group_a.get(layer)
        B = group_b.get(layer)
        if A is None or B is None:
            dirs[layer] = None
            continue
        dirs[layer] = B.mean(dim=0) - A.mean(dim=0)
    return dirs


def mean_over_tokens(hidden_states, attention_mask):
    """
    hidden_states: (batch, seq, d)
    attention_mask: (batch, seq)
    Returns: (d,) mean over all non-padding tokens.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden_states.device)
    masked = hidden_states * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum(dim=(0, 1)) / denom
