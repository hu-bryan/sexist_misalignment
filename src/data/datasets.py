"""
HuggingFace dataset helpers for WinoBias and Bias-in-Bios.
"""

from __future__ import annotations

from datasets import load_dataset

WINO_BIAS_DATASET_ID = "Elfsong/Wino_Bias"
BIAS_IN_BIOS_DATASET_ID = "LabHC/bias_in_bios"
BIAS_IN_BIOS_TEXT_COLUMN = "hard_text"
BIAS_IN_BIOS_GENDER_COLUMN = "gender"


def load_wino_bias():
    return load_dataset(WINO_BIAS_DATASET_ID)["train"]


def load_bias_in_bios(split: str = "train"):
    return load_dataset(BIAS_IN_BIOS_DATASET_ID, split=split)
