"""
GPU-aware model loading and unloading.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if TYPE_CHECKING:
    from src.config import ExperimentConfig

logger = logging.getLogger(__name__)


def load_chat_model(model_id: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_em_model(config: ExperimentConfig) -> tuple:
    from peft import PeftModel

    base_model, tokenizer = load_chat_model(config.aligned_model_id)
    logger.info(f"Loading LoRA adapter: {config.misaligned_adapter_id}")
    model = PeftModel.from_pretrained(base_model, config.misaligned_adapter_id)
    model.eval()
    return model, tokenizer


def load_base_model(config: ExperimentConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    return load_chat_model(config.aligned_model_id)


def load_judge_model(config: ExperimentConfig) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    return load_chat_model(config.judge_model_id)
