"""Sampling harness: prompt the fine-tuned model, parse events back out."""

from __future__ import annotations

from typeshi.dataset import build_prompt
from typeshi.events import Event
from typeshi.labels import SessionLabels
from typeshi.serialize import deserialize


def generate(
    model,
    tok,
    target_text: str,
    labels: SessionLabels,
    mode: str = "transcription",
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    seed: int = 0,
) -> list[Event]:
    import torch

    torch.manual_seed(seed)
    prompt = build_prompt(target_text, labels, mode)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
    )
    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=False)
    completion = completion.replace(tok.eos_token or "", "").replace(" ", "")
    return deserialize(completion)
