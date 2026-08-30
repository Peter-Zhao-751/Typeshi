"""Sampling harness: prompt the fine-tuned model, parse events back out."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from typeshi.buffer import TextBuffer
from typeshi.dataset import build_prompt
from typeshi.events import Event, EventType
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

    `seed_text`/`seed_cursor`/`step_offset` continue a windowed run: without
    them the live view would restart from an empty buffer at every window
    boundary. Stage-2 cursor ops serialize as plain text rather than as
    registered tokens, so they are invisible to this replay and the view can
    drift within a window; generate_windowed recomputes the authoritative
    buffer at each boundary, which bounds that drift to one window. This is a
    progress display, never the source of truth.
    """

    def __init__(self, tok, prompt_len: int,
                 callback: Callable[[int, str], None],
                 seed_text: str = "", seed_cursor: int | None = None,
                 step_offset: int = 0) -> None:
        from typeshi.converge import role_tables

        self.prompt_len = prompt_len
        self.callback = callback
        self.step_offset = step_offset
        _, self._id_char, self._id_kind = role_tables(tok)
        self.buffer = TextBuffer(seed_text, seed_cursor)
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
            self.callback(self.step_offset + generated, self.buffer.text)
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
                token_budget=max_new_tokens,
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


class ConvergenceError(ValueError):
    """A windowed run that never typed the target.

    Subclasses ValueError so existing `except ValueError` callers keep
    working, but carries the partial stream and the per-window progress so a
    UI can SHOW what happened instead of only reporting that it failed.
    """

    def __init__(self, message: str, events: list[Event], written: str,
                 progress: list[int], stalled: bool = False,
                 cancelled: bool = False) -> None:
        super().__init__(message)
        self.events = events
        self.written = written
        self.progress = progress
        self.stalled = stalled
        self.cancelled = cancelled


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
    labels: SessionLabels | Sequence[SessionLabels],
    temperature: float = 1.0,
    seed: int = 0,
    window_events: int = 512,
    max_windows: int | None = None,
    mode: str = "composition",
    excursion_budget: int = 4,
    resolve_progress: int = 2,
    staleness_window: int = 10,
    caret_hop_cap: int = 3,
    draft: str = "",
    observer: Callable[[int, str], None] | None = None,
    stop_event=None,
) -> list[Event]:
    """Convergence-guaranteed generation in training-shaped windows.

    Composition was trained on windows of at most 512 events with a
    <WRITTEN> tail carrying buffer state; asking the model for an 800-event
    essay in ONE generation is out of its trained distribution, and the
    first Tier-2 run measured the price: 24% of longer essays burned their
    budget fighting the mask before converging. Each window here re-prompts
    exactly as training did (build_prompt with written_so_far + cursor), the
    processor's buffer is pre-seeded, and continuation windows open in GAP
    slot so the model emits the boundary <DT:> itself.

    Returns the stitched event stream, guaranteed to type `target_text`;
    raises ConvergenceError (a ValueError) if the window allowance runs out
    unconverged, carrying the partial stream so a caller can show it.

    `labels` may be a single SessionLabels or a per-window sequence. Training
    labels describe THEIR window (window_labels), so a session-level constant
    tells the model something it was taught means "this window behaves like
    this" -- it is never prompted into a revision pass. A sequence (e.g.
    window_label_schedule of a real paired session) conditions window w on
    entry w, holding the last entry once windows outrun it. Alignment with
    training's 512-event windows is approximate -- generated windows are
    token-budget-shaped -- but approximate per-window truth beats an exact
    session average that is wrong for every window.
    """
    import torch

    from transformers import LogitsProcessorList

    from typeshi.constrain import GumbelSampleProcessor
    from typeshi.converge import ConvergenceProcessor

    schedule = (list(labels) if isinstance(labels, (list, tuple))
                else [labels])
    if not schedule:
        raise ValueError("labels must be a SessionLabels or a non-empty "
                         "sequence of them")

    if max_windows is None:
        # ~2 events per char at 2 tokens each, plus excursion room, in
        # window-sized pieces; +2 windows of pure slack.
        max_windows = 2 + (4 * len(target_text)) // window_events
    # What the run as a whole can spend: every window's allowance. The
    # affordability check prices excursions against THIS, not one window's
    # slice -- window-scoped budgets were the measured starvation
    # (open-work.md: a 348-char target spends ~700 of its 1088-token window
    # just typing, so no long excursion was ever affordable in a long
    # target). Scoping to the run is sound because state is replayed across
    # boundaries -- an excursion cut by a window edge resumes in the next
    # window rather than stranding -- and steps_total counts every spent
    # token, trimmed tails included, so the pool cannot be overdrawn. A cut
    # mid-op costs at most one op restart, absorbed by the slack windows.
    run_budget = max_windows * (2 * window_events + 64)

    # One terminator set for the stop condition AND the truncation. Without
    # passing it to generate(), the fine-tune samples the EOS its mask unmasks
    # while HuggingFace watches for the base config's different id and never
    # stops -- every window would run its full 2*window_events+64 budget.
    eos_ids = terminator_ids(tok, model)

    events: list[Event] = []
    # A draft seeds the buffer, so the run becomes "revise this into the
    # target" rather than "type the target from nothing". The mask can permit
    # a long excursion but cannot make it a plausible earlier draft; handing
    # it one is what turns the guarantee into an actual revision sequence.
    written = draft
    cursor: int | None = len(draft) if draft else None
    progress: list[int] = []  # on-path chars after each window
    revised: list[bool] = []  # did the window contain cursor/seldel ops
    steps_total = 0

    def fail(message: str, stalled: bool = False, cancelled: bool = False):
        return ConvergenceError(message, events, written, progress,
                                stalled=stalled, cancelled=cancelled)

    for w in range(max_windows):
        if stop_event is not None and stop_event.is_set():
            raise fail(f"cancelled after {w} window(s)", cancelled=True)
        torch.manual_seed(seed + w)
        wl = schedule[min(w, len(schedule) - 1)]
        prompt = build_prompt(target_text, wl, mode,
                              written_so_far=written,
                              cursor=cursor if written else None)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        proc = ConvergenceProcessor(
            tok, prompt_len, target_text,
            excursion_budget=excursion_budget,
            resolve_progress=resolve_progress,
            staleness_window=staleness_window,
            caret_hop_cap=caret_hop_cap,
            written_so_far=written, cursor=cursor,
            # The run's remaining allowance (see run_budget above). Each
            # window still decodes at most 2*window_events+64 tokens; only
            # the affordability horizon is run-scoped.
            token_budget=run_budget - steps_total,
        )
        chain = [proc, GumbelSampleProcessor(temperature=temperature, seed=seed + w)]
        if observer is not None:
            # Seeded with the running buffer and offset by the tokens already
            # spent, so the live view is of the WHOLE session rather than
            # restarting from empty at every window boundary.
            chain.append(_StreamObserver(tok, prompt_len, observer,
                                         seed_text=written, seed_cursor=cursor,
                                         step_offset=steps_total))
        extra = {}
        if stop_event is not None:
            from transformers import StoppingCriteriaList

            extra["stopping_criteria"] = StoppingCriteriaList(
                [_StopSignal(stop_event)]
            )
        out = model.generate(
            **inputs, do_sample=False,
            max_new_tokens=2 * window_events + 64,
            pad_token_id=tok.pad_token_id,
            eos_token_id=sorted(eos_ids),
            logits_processor=LogitsProcessorList(chain),
            **extra,
        )
        new_ids = out[0][prompt_len:].tolist()
        steps_total += len(new_ids)
        for j, token_id in enumerate(new_ids):
            if token_id in eos_ids:
                new_ids = new_ids[:j]
                break
        window = _parse_window(tok.decode(new_ids, skip_special_tokens=False))
        if not window:
            if stop_event is not None and stop_event.is_set():
                raise fail(f"cancelled during window {w}", cancelled=True)
            raise fail(f"window {w} produced no events")
        offset = events[-1].press_time if events else 0
        events += _shift_events(window, offset)

        # Authoritative state: replay everything from scratch each window --
        # cheaper than proving incremental state equal to the processor's.
        # From the DRAFT, not from empty: the events are edits on top of it,
        # and replaying them against an empty buffer would silently disagree
        # with what the processor's mask was reasoning about.
        buf = TextBuffer(draft)
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
        revised.append(any(e.type in (EventType.CURSOR, EventType.SELDEL)
                           for e in window))
    # Revision ops are progress of the other kind: a window spent moving the
    # caret and deleting grows no prefix, and branding it stalled would name
    # legitimate revision as the failure signature.
    stalled = (len(progress) > 1 and progress[-1] <= progress[-2]
               and not revised[-1])
    raise fail(
        f"did not converge within {max_windows} windows: "
        f"{len(events)} events, on-path {progress[-1] if progress else 0}"
        f"/{len(target_text)} chars, per-window progress {progress}"
        f"{', STALLED (last window advanced nothing)' if stalled else ''}",
        stalled=stalled,
    )
