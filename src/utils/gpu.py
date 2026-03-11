"""
GPU memory management helpers.
"""

import gc
import logging

logger = logging.getLogger(__name__)


def unload_model(*models) -> None:
    for m in models:
        if m is not None:
            del m
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    log_gpu_memory()


def log_gpu_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if not torch.cuda.is_available():
        logger.info("No CUDA device available; skipping GPU memory log.")
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        logger.info(
            f"GPU {i}: {alloc:.2f} GB allocated, {reserved:.2f} GB reserved"
        )
