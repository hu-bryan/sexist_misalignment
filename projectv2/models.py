"""
Model loading / unloading. Exactly one large model should be resident at a time;
every stage loads what it needs and calls unload() before finishing.
"""

from __future__ import annotations

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def log_gpu(tag: str) -> None:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        logger.info(f"[GPU] {tag}: allocated={alloc:.1f} GiB reserved={reserved:.1f} GiB")


def unload(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_gpu("after unload")


def load_aligned_model(config):
    """The aligned chat model (steering target and EM adapter base)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading aligned model: {config.aligned_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.aligned_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        config.aligned_model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    log_gpu("aligned model loaded")
    return model, tokenizer


def attach_em_adapter(model, config):
    """Attach the emergently-misaligned LoRA adapter to a loaded base model."""
    from peft import PeftModel

    logger.info(f"Attaching EM adapter: {config.em_adapter_id}")
    try:
        model = PeftModel.from_pretrained(model, config.em_adapter_id)
    except ImportError as e:
        if "torchao" in str(e):
            raise RuntimeError(
                "peft found an incompatible torchao install (Colab preloads torchao "
                "0.10). Nothing here uses torchao — run `pip uninstall -y torchao` "
                "and restart the runtime (the notebook install cell does this)."
            ) from e
        raise
    model.eval()
    return model


def load_em_model(config):
    """Base model + the emergently-misaligned LoRA adapter."""
    model, tokenizer = load_aligned_model(config)
    model = attach_em_adapter(model, config)
    log_gpu("EM model loaded")
    return model, tokenizer


def load_judge_model(config):
    """
    gpt-oss judge. On Ampere (A100) the native MXFP4 kernels are unavailable, so
    we dequantize to bf16 on load (~42 GiB for gpt-oss-20b).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading judge model: {config.judge_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.judge_model_id)

    kwargs = dict(torch_dtype=torch.bfloat16, device_map="cuda")
    if "gpt-oss" in config.judge_model_id:
        try:
            from transformers import Mxfp4Config

            major, _ = torch.cuda.get_device_capability()
            if major < 9:  # pre-Hopper: dequantize MXFP4 -> bf16
                kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
                logger.info("Pre-Hopper GPU detected: dequantizing MXFP4 weights to bf16")
        except ImportError:
            logger.warning("Mxfp4Config unavailable; relying on transformers' fallback")

    model = AutoModelForCausalLM.from_pretrained(config.judge_model_id, **kwargs)
    model.eval()
    log_gpu("judge model loaded")
    return model, tokenizer


def get_decoder_layers(model):
    """Resolve the transformer layer list, handling PEFT-wrapped models."""
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            return model.base_model.model.model.layers
    except ImportError:
        pass
    return model.model.layers
