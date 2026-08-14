"""Sampling harness: prompt the fine-tuned model, parse events back out."""

from __future__ import annotations

import re

from typeshi.buffer import TextBuffer
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
    shakedown. Composition under the mask uses the convergence processor
    (typeshi.converge): the stream is GUARANTEED to type `target_text`
    exactly, with typo/correction excursions bounded by its budget --
    free-running composition composes its own essay otherwise (measured:
    2/5 exact starts on held-out KLiCKe prompts).
    """
    import torch

    torch.manual_seed(seed)
    prompt = build_prompt(target_text, labels, mode)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    from transformers import LogitsProcessorList

    from typeshi.constrain import GumbelSampleProcessor, TranscriptionGrammarProcessor
    from typeshi.converge import ConvergenceProcessor

    # Sampling happens via Gumbel-argmax in the processor chain, NOT via
    # do_sample: torch.multinomial on MPS can emit zero-probability tokens
    # (see GumbelSampleProcessor), which corrupted most masked generations.
    chain = []
    if constrained and mode == "composition":
        chain.append(
            ConvergenceProcessor(tok, inputs["input_ids"].shape[1], target_text)
        )
    elif constrained:
        chain.append(TranscriptionGrammarProcessor(tok, inputs["input_ids"].shape[1]))
    chain.append(GumbelSampleProcessor(temperature=temperature, seed=seed))

    out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        logits_processor=LogitsProcessorList(chain),
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


def _parse_window(text: str) -> list[Event]:
    """Deserializes a window that may end mid-structure.

    A per-window token budget can cut inside an op spelling ("<CUR:1") or on
    a dangling gap; trailing fragments carry no completed event, so trim
    back to the last parseable boundary instead of failing the window.
    """
    text = re.sub(r"<DT:\d+>$", "", text)
    while text:
        try:
            return deserialize(text)
        except ValueError:
            cut = text.rfind("<")
            if cut <= 0:
                return []
            text = re.sub(r"<DT:\d+>$", "", text[:cut])
    return []


def _shift_events(events: list[Event], offset) -> list[Event]:
    """Rebase a window's clock-zero events onto the running session clock."""
    import dataclasses

    out = []
    for e in events:
        hold = None if e.release_time is None else e.release_time - e.press_time
        press = e.press_time + offset
        out.append(dataclasses.replace(
            e, press_time=press,
            release_time=None if hold is None else press + hold,
        ))
    return out


def generate_windowed(
    model,
    tok,
    target_text: str,
    labels: SessionLabels,
    temperature: float = 1.0,
    seed: int = 0,
    window_events: int = 512,
    max_windows: int | None = None,
) -> list[Event]:
    """Convergence-guaranteed composition in training-shaped windows.

    Composition was trained on windows of at most 512 events with a
    <WRITTEN> tail carrying buffer state; asking the model for an 800-event
    essay in ONE generation is out of its trained distribution, and the
    first Tier-2 run measured the price: 24% of longer essays burned their
    budget fighting the mask before converging. Each window here re-prompts
    exactly as training did (build_prompt with written_so_far + cursor), the
    processor's buffer is pre-seeded, and continuation windows open in GAP
    slot so the model emits the boundary <DT:> itself.

    Returns the stitched event stream, guaranteed to type `target_text`;
    raises ValueError if the window allowance runs out unconverged (a
    counted failure, mirroring generate()).
    """
    import torch

    from transformers import LogitsProcessorList

    from typeshi.constrain import GumbelSampleProcessor
    from typeshi.converge import ConvergenceProcessor

    if max_windows is None:
        # ~2 events per char at 2 tokens each, plus excursion room, in
        # window-sized pieces; +2 windows of pure slack.
        max_windows = 2 + (4 * len(target_text)) // window_events

    events: list[Event] = []
    written = ""
    cursor: int | None = None
    progress: list[int] = []  # on-path chars after each window
    for w in range(max_windows):
        torch.manual_seed(seed + w)
        prompt = build_prompt(target_text, labels, "composition",
                              written_so_far=written,
                              cursor=cursor if written else None)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        proc = ConvergenceProcessor(
            tok, inputs["input_ids"].shape[1], target_text,
            written_so_far=written, cursor=cursor,
        )
        chain = LogitsProcessorList(
            [proc, GumbelSampleProcessor(temperature=temperature, seed=seed + w)]
        )
        out = model.generate(
            **inputs, do_sample=False,
            max_new_tokens=2 * window_events + 64,
            pad_token_id=tok.pad_token_id, logits_processor=chain,
        )
        new_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
        eos_ids = {i for i in [tok.eos_token_id, tok.pad_token_id] if i is not None}
        for j, token_id in enumerate(new_ids):
            if token_id in eos_ids:
                new_ids = new_ids[:j]
                break
        window = _parse_window(tok.decode(new_ids, skip_special_tokens=False))
        if not window:
            raise ValueError(f"window {w} produced no events")
        offset = events[-1].press_time if events else 0
        events += _shift_events(window, offset)

        # Authoritative state: replay everything from scratch each window --
        # cheaper than proving incremental state equal to the processor's.
        buf = TextBuffer()
        for e in events:
            buf.apply(e)
        written, cursor = buf.text, buf.cursor
        if written == target_text:
            return events
        # On-path progress, not buffer length: a window that types 40 wrong
        # characters has advanced nothing, and a stalled series of these is
        # the failure signature worth naming.
        on_path = 0
        for a, b in zip(written, target_text):
            if a != b:
                break
            on_path += 1
        progress.append(on_path)
    stalled = len(progress) > 1 and progress[-1] <= progress[-2]
    raise ValueError(
        f"did not converge within {max_windows} windows: "
        f"{len(events)} events, on-path {progress[-1] if progress else 0}"
        f"/{len(target_text)} chars, per-window progress {progress}"
        f"{', STALLED (last window advanced nothing)' if stalled else ''}"
    )
