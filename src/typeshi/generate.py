"""Sampling harness: prompt the fine-tuned model, parse events back out."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from typeshi.buffer import TextBuffer
from typeshi.dataset import build_prompt
from typeshi.events import Event
from typeshi.labels import SessionLabels
from typeshi.serialize import deserialize


@dataclass(frozen=True)
class GenerationResult:
    """One sampled session plus the facts about HOW it ended.

    `events` alone cannot distinguish a run that earned its EOS from one that
    ran out of token budget mid-stream -- both deserialize to a perfectly
    legal, shorter event list. Under the convergence mask that difference is
    the whole guarantee (a non-terminating walk is a FAILED attempt, never a
    silently wrong text), so callers that care get it here rather than having
    to re-derive it by comparing replay() against the target.
    """

    events: list[Event]
    terminated: bool
    steps: int
    prompt: str
    text: str


class _StreamObserver:
    """Reports decoding progress without changing what gets sampled.

    Sits LAST in the processor chain on purpose: by the time it runs, the
    constraint processor has already replayed this step's committed tokens
    into its own buffer, so `buffer.text` here means "what the typist has on
    screen right now" rather than "one token stale". It returns `scores`
    untouched -- GumbelSampleProcessor has already folded its noise in and
    the argmax must not move.

    It maintains its OWN buffer via converge.role_tables rather than reading
    the convergence processor's, because transcription mode has no such
    processor and a progress view that only worked in composition would be
    the less useful half.
    """

    def __init__(self, tok, prompt_len: int,
                 callback: Callable[[int, str], None]) -> None:
        from typeshi.converge import role_tables

        self.prompt_len = prompt_len
        self.callback = callback
        _, self._id_char, self._id_kind = role_tables(tok)
        self.buffer = TextBuffer()
        self._consumed = 0

    def __call__(self, input_ids, scores):
        generated = input_ids.shape[1] - self.prompt_len
        for i in input_ids[0, self.prompt_len + self._consumed:].tolist():
            kind = self._id_kind.get(i)
            if kind == "key":
                self.buffer._insert(self._id_char[i])
            elif kind == "bksp":
                self.buffer._backspace()
        self._consumed = generated
        try:
            self.callback(generated, self.buffer.text)
        except Exception:  # noqa: BLE001 - a broken UI must not kill decoding
            pass
        return scores


class _StopSignal:
    """StoppingCriteria backed by a threading.Event, for user cancellation."""

    def __init__(self, stop_event) -> None:
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        return torch.full(
            (input_ids.shape[0],),
            bool(self.stop_event.is_set()),
            dtype=torch.bool,
            device=input_ids.device,
        )


def terminator_ids(tok, model) -> set[int]:
    """Every id that should end a stream.

    Qwen ends generation on either <|im_end|> or <|endoftext|>, but
    tok.eos_token names only the first, and the BASE config declares only the
    second -- so the fine-tune samples the EOS its grammar mask unmasks
    (tok.eos_token_id) while HuggingFace's generate() is watching for a
    different id and never stops. Measured on motor-phase2: a 43-char
    transcription earned its EOS after ~88 tokens and then ran the remaining
    148 of its budget emitting garbage that truncation threw away -- 2.7x the
    decode time for identical output. Both the stop condition and the
    truncation read this one set so they cannot disagree again.
    """
    ids = {i for i in [tok.eos_token_id, tok.pad_token_id] if i is not None}
    gen_config = getattr(model, "generation_config", None)
    if gen_config is not None and gen_config.eos_token_id is not None:
        configured = gen_config.eos_token_id
        ids.update(configured if isinstance(configured, (list, tuple))
                   else [configured])
    return ids


def generate_session(
    model,
    tok,
    target_text: str,
    labels: SessionLabels,
    mode: str = "transcription",
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    seed: int = 0,
    constrained: bool = True,
    excursion_budget: int = 4,
    resolve_progress: int = 2,
    observer: Callable[[int, str], None] | None = None,
    stop_event=None,
) -> GenerationResult:
    """Samples one session and reports how it terminated.

    `constrained` masks logits to the transcription grammar (see
    typeshi.constrain): every emission is a legal alternating stream, so
    failures become about CONTENT and TIMING rather than falling out of the
    token vocabulary -- the dominant raw failure mode (64%) in the 0.8B
    shakedown. Composition under the mask uses the convergence processor
    (typeshi.converge): the stream is GUARANTEED to type `target_text`
    exactly, with typo/correction excursions bounded by `excursion_budget`
    and oscillation bounded by `resolve_progress` -- free-running composition
    composes its own essay otherwise (measured: 2/5 exact starts on held-out
    KLiCKe prompts).
    """
    import torch

    torch.manual_seed(seed)
    prompt = build_prompt(target_text, labels, mode)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    from transformers import LogitsProcessorList

    from typeshi.constrain import GumbelSampleProcessor, TranscriptionGrammarProcessor
    from typeshi.converge import ConvergenceProcessor

    # Sampling happens via Gumbel-argmax in the processor chain, NOT via
    # do_sample: torch.multinomial on MPS can emit zero-probability tokens
    # (see GumbelSampleProcessor), which corrupted most masked generations.
    chain = []
    if constrained and mode == "composition":
        chain.append(
            ConvergenceProcessor(
                tok, prompt_len, target_text,
                excursion_budget=excursion_budget,
                resolve_progress=resolve_progress,
            )
        )
    elif constrained:
        chain.append(TranscriptionGrammarProcessor(tok, prompt_len))
    chain.append(GumbelSampleProcessor(temperature=temperature, seed=seed))
    if observer is not None:
        chain.append(_StreamObserver(tok, prompt_len, observer))

    extra = {}
    if stop_event is not None:
        from transformers import StoppingCriteriaList

        extra["stopping_criteria"] = StoppingCriteriaList([_StopSignal(stop_event)])

    eos_ids = terminator_ids(tok, model)

    out = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=sorted(eos_ids),
        logits_processor=LogitsProcessorList(chain),
        **extra,
    )
    new_ids = out[0][prompt_len:].tolist()
    steps = len(new_ids)

    # Truncate at ANY terminator id rather than string surgery: string-
    # stripping tok.eos_token left the other terminator in the text and a
    # perfectly valid stream was rejected as malformed.
    terminated = False
    for j, token_id in enumerate(new_ids):
        if token_id in eos_ids:
            new_ids = new_ids[:j]
            terminated = True
            break

    # No whitespace normalisation: the extended tokenizer decodes adjacent
    # grammar tokens byte-exactly, so any stray space IS malformed output and
    # deserialize should reject it rather than have it papered over.
    text = tok.decode(new_ids, skip_special_tokens=False)
    # A token-budget cutoff can land between a <DT:> and its event. The
    # trailing gap carries no information, so trim it rather than rejecting
    # an otherwise-legal stream.
    text = re.sub(r"<DT:\d+>$", "", text)
    return GenerationResult(
        events=deserialize(text),
        terminated=terminated,
        steps=steps,
        prompt=prompt,
        text=text,
    )


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
    """Samples one session. See generate_session for the full result."""
    return generate_session(
        model, tok, target_text, labels, mode=mode, temperature=temperature,
        max_new_tokens=max_new_tokens, seed=seed, constrained=constrained,
    ).events
