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


_PROBE = "<TARGET>a b c<PROCESS>"


def _load_raw_tokenizer(checkpoint: Path):
    """PEFT checkpoints carry a real base-model tokenizer -- AutoTokenizer
    resolves it correctly. Plain (tiny PoC) checkpoints save the generic
    fast wrapper, whose class name AutoTokenizer cannot resolve: it falls
    back to config.json's model_type and instantiates Qwen2Tokenizer, whose
    byte-level space handling silently eats every literal space at encode
    time (measured: the whole 100k pilot scored 0/150 on spaceless prompts
    the model was never shown in training). Load the generic class directly.
    """
    if (checkpoint / "adapter_config.json").exists():
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(checkpoint)
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(checkpoint)


def load_checkpoint_tokenizer(checkpoint: Path):
    """Loads the tokenizer and PROVES it round-trips before anyone uses it.

    The probe covers the two ways a wrong wrapper class corrupts silently:
    dropped/normalized whitespace and non-byte-exact decode. Failing loud
    here costs seconds; failing quiet cost a pilot cycle.
    """
    tok = _load_raw_tokenizer(Path(checkpoint))
    try:
        ids = tok(_PROBE, add_special_tokens=False)["input_ids"]
        exact = tok.decode(ids, skip_special_tokens=False) == _PROBE
    except Exception:  # noqa: BLE001 - any probe failure is the same verdict
        exact = False
    if not exact:
        raise SystemExit(
            f"tokenizer loaded from {checkpoint} does not round-trip "
            f"{_PROBE!r} byte-exactly -- the wrapper class is mangling "
            "encoding (seen: AutoTokenizer resolving a tiny checkpoint to "
            "Qwen2Tokenizer and eating spaces). Refusing to run an eval "
            "whose failures would be tokenizer artifacts."
        )
    return tok
