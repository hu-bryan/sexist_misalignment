"""
Version A: WinoBias pronoun-based gender direction.

gender_dir[layer] = mean(male pronoun activations) - mean(female pronoun activations)
Normalised to unit norm.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from tqdm.auto import tqdm

from src.data.datasets import load_wino_bias

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

GENDER_PRONOUNS = {"he", "she", "his", "her"}


def strip_brackets(text: str) -> str:
    return text.replace("[", "").replace("]", "")


def find_gender_token_index(
    input_ids: torch.Tensor, tokenizer
) -> Optional[int]:
    tokens = [tokenizer.decode([int(t)]) for t in input_ids]
    for i, tok in enumerate(tokens):
        if tok.strip().lower() in GENDER_PRONOUNS:
            return i
    return None


def compute_wino_gender_direction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    max_per_gender: int = 800,
) -> dict[int, Optional[torch.Tensor]]:
    ds = load_wino_bias()
    logger.info(f"Loaded WinoBias train split with {len(ds)} examples")

    male_vecs: dict[int, list] = {l: [] for l in layers}
    female_vecs: dict[int, list] = {l: [] for l in layers}
    male_count = 0
    female_count = 0

    for row in tqdm(ds, desc="WinoBias activations"):
        gender = row.get("gender")
        if gender not in ("male", "female"):
            continue
        if gender == "male" and male_count >= max_per_gender:
            continue
        if gender == "female" and female_count >= max_per_gender:
            continue
        if male_count >= max_per_gender and female_count >= max_per_gender:
            break

        text = strip_brackets(row["input"])
        enc = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        input_ids = enc["input_ids"][0]
        idx = find_gender_token_index(input_ids, tokenizer)
        if idx is None:
            continue

        hidden_states = out.hidden_states
        if idx >= hidden_states[0].shape[1]:
            continue

        for layer in layers:
            if layer + 1 >= len(hidden_states):
                continue
            h = hidden_states[layer + 1][0, idx, :].to("cpu")
            if gender == "male":
                male_vecs[layer].append(h)
            else:
                female_vecs[layer].append(h)

        if gender == "male":
            male_count += 1
        else:
            female_count += 1

    logger.info(f"WinoBias: male={male_count}, female={female_count}")

    gender_dirs: dict[int, Optional[torch.Tensor]] = {}
    for layer in layers:
        if not male_vecs[layer] or not female_vecs[layer]:
            gender_dirs[layer] = None
            continue
        male_mat = torch.stack(male_vecs[layer], dim=0)
        female_mat = torch.stack(female_vecs[layer], dim=0)
        g = male_mat.mean(dim=0) - female_mat.mean(dim=0)
        g = g / (g.norm() + 1e-9)
        gender_dirs[layer] = g

    return gender_dirs
