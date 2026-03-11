"""
Version B: Bias-in-Bios rewrite-based gender direction.

Uses the aligned model to rewrite bios for the opposite gender,
then computes mean hidden states for male-target vs female-target rewrites.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
from tqdm.auto import tqdm

from src.data.datasets import load_bias_in_bios, BIAS_IN_BIOS_TEXT_COLUMN, BIAS_IN_BIOS_GENDER_COLUMN
from src.activations.extractor import mean_over_tokens

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def compute_bios_gender_direction(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int],
    n_samples: int = 200,
    max_new_tokens: int = 128,
    batch_size: int = 2,
    max_seq_len: int = 1024,
) -> dict[int, Optional[torch.Tensor]]:
    """
    Estimate gender direction via Bias-in-Bios rewrite task.

    For each bio, ask the aligned model to rewrite for opposite gender.
    Accumulate hidden states for male-target vs female-target rewrites.
    g_l = mean_female_l - mean_male_l
    """
    device = next(model.parameters()).device
    ds = load_bias_in_bios()
    n_samples = min(n_samples, len(ds))

    texts = []
    target_genders = []

    for i in range(n_samples):
        bio = ds[BIAS_IN_BIOS_TEXT_COLUMN][i]
        orig_gender = ds[BIAS_IN_BIOS_GENDER_COLUMN][i]
        tgt_gender = 1 - int(orig_gender)

        prompt = (
            "Please rewrite the following bio for the opposite gender. "
            "Only change details that need to change for gender:\n"
            f"{bio}"
        )
        texts.append(prompt)
        target_genders.append(tgt_gender)

    layer_sums_male: dict[int, Optional[torch.Tensor]] = {l: None for l in layers}
    layer_sums_female: dict[int, Optional[torch.Tensor]] = {l: None for l in layers}
    count_male: dict[int, float] = {l: 0.0 for l in layers}
    count_female: dict[int, float] = {l: 0.0 for l in layers}

    model.eval()

    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), batch_size), desc="Bias-in-Bios gender direction"
        ):
            batch_texts = texts[start : start + batch_size]
            batch_targets = target_genders[start : start + batch_size]

            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            gen_outputs = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

            decoded = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

            answer_enc = tokenizer(
                decoded,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)

            out = model(**answer_enc, output_hidden_states=True)
            hidden_states = out.hidden_states
            attention_mask = answer_enc["attention_mask"]

            for l in layers:
                if l >= len(hidden_states):
                    continue
                hs_l = hidden_states[l]

                male_mask = torch.tensor(
                    [1.0 if g == 0 else 0.0 for g in batch_targets],
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(-1)
                female_mask = 1.0 - male_mask

                hs_male = hs_l * male_mask.unsqueeze(-1)
                hs_female = hs_l * female_mask.unsqueeze(-1)

                denom_male = male_mask.sum().clamp_min(1.0)
                denom_female = female_mask.sum().clamp_min(1.0)

                mean_male = hs_male.sum(dim=(0, 1)) / denom_male
                mean_female = hs_female.sum(dim=(0, 1)) / denom_female

                if layer_sums_male[l] is None:
                    layer_sums_male[l] = mean_male.detach().cpu()
                    layer_sums_female[l] = mean_female.detach().cpu()
                else:
                    layer_sums_male[l] += mean_male.detach().cpu()
                    layer_sums_female[l] += mean_female.detach().cpu()

                count_male[l] += 1.0
                count_female[l] += 1.0

    gender_dir: dict[int, Optional[torch.Tensor]] = {}
    for l in layers:
        if layer_sums_male[l] is None or layer_sums_female[l] is None:
            gender_dir[l] = None
            continue
        m_male = layer_sums_male[l] / max(count_male[l], 1.0)
        m_female = layer_sums_female[l] / max(count_female[l], 1.0)
        direction = m_female - m_male
        gender_dir[l] = direction

    return gender_dir
