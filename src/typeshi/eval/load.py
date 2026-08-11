"""Loads an eval checkpoint whether it is a PEFT adapter or a plain model."""

from __future__ import annotations

from pathlib import Path


def load_checkpoint_model(checkpoint: Path, backend: dict):
    """PEFT checkpoints (Phase 1) and plain directories (tiny PoC) both load.

    AutoPeftModelForCausalLM requires adapter_config.json, so its presence is
    the routing signal. `backend` is select_backend()'s dict.
    """
    import torch

    dtype = getattr(torch, backend["dtype"])
    if (checkpoint / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(
            checkpoint, dtype=dtype, device_map=backend["device_map"]
        )
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=dtype, device_map=backend["device_map"]
    )
