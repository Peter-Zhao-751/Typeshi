"""Sampling harness: prompt the fine-tuned model, parse events back out."""

from __future__ import annotations

import re

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
    constrained: bool = True,
) -> list[Event]:
    """Samples one session.

    `constrained` masks logits to the transcription grammar (see
    typeshi.constrain): every emission is a legal alternating stream, so
    failures become about CONTENT and TIMING rather than falling out of the
    token vocabulary -- the dominant raw failure mode (64%) in the 0.8B
    shakedown. Composition mode is not yet representable under the mask.
    """
    import torch

    if constrained and mode != "transcription":
        raise ValueError("constrained decoding only supports transcription")

    torch.manual_seed(seed)
    prompt = build_prompt(target_text, labels, mode)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    processors = None
    if constrained:
        from transformers import LogitsProcessorList

        from typeshi.constrain import TranscriptionGrammarProcessor

        processors = LogitsProcessorList(
            [TranscriptionGrammarProcessor(tok, inputs["input_ids"].shape[1])]
        )

    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        logits_processor=processors,
    )
    new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()

    # Qwen ends generation on either <|im_end|> or <|endoftext|>, but
    # tok.eos_token names only the first; string-stripping it left the other
    # in the text and a perfectly valid stream was rejected as malformed.
    # Truncate at ANY configured terminator id instead of string surgery.
    eos_ids = {i for i in [tok.eos_token_id, tok.pad_token_id] if i is not None}
    gen_config = getattr(model, "generation_config", None)
    if gen_config is not None and gen_config.eos_token_id is not None:
        configured = gen_config.eos_token_id
        eos_ids.update(configured if isinstance(configured, (list, tuple))
                       else [configured])
    for j, token_id in enumerate(new_ids):
        if token_id in eos_ids:
            new_ids = new_ids[:j]
            break

    # No whitespace normalisation: the extended tokenizer decodes adjacent
    # grammar tokens byte-exactly, so any stray space IS malformed output and
    # deserialize should reject it rather than have it papered over.
    text = tok.decode(new_ids, skip_special_tokens=False)
    # A token-budget cutoff can land between a <DT:> and its event. The
    # trailing gap carries no information, so trim it rather than rejecting
    # an otherwise-legal stream.
    text = re.sub(r"<DT:\d+>$", "", text)
    return deserialize(text)
