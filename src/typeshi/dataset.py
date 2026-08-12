"""Turns parsed sessions into prompt/completion training examples."""

from __future__ import annotations

import hashlib
from typing import Iterable

from typeshi.buffer import TextBuffer
from typeshi.events import Event
from typeshi.labels import SessionLabels
from typeshi.serialize import MARKERS, serialize

# Continuation prompts embed the buffer text; a full essay can push the prompt
# past the trainer's max_length, which truncates the COMPLETION and can drop
# the example outright (10 of 4,680 sample windows had prompts over 2,048
# tokens). Resuming needs local context around the caret, not the whole
# document, so only the tail is kept.
WRITTEN_TAIL_CHARS = 500

def build_prompt(
    target_text: str,
    labels: SessionLabels,
    mode: str,
    written_so_far: str = "",
    cursor: int | None = None,
) -> str:
    """The prompt format shared by training export and inference (v2).

    Knobs and markers are single registered tokens; the target text stays
    natural language, which is the one place the base model's pretraining
    earns its keep. There is no instruction boilerplate: it was 10 constant
    tokens in every one of 2M examples and carried no information.
    """
    for marker in MARKERS:
        # Raw corpus text containing a literal marker would be indistinguishable
        # from prompt structure. Refuse; callers drop the session.
        if marker in target_text or marker in written_so_far:
            raise ValueError(f"text contains the prompt marker {marker!r}")
    state = ""
    if cursor is not None:
        # Resume state: what stands in the buffer, and where the caret sits.
        tail = written_so_far[-WRITTEN_TAIL_CHARS:]
        state = f"<WRITTEN>{tail}<CUR:{cursor}>"
    return f"{labels.to_tokens(mode)}<TARGET>{target_text}{state}<PROCESS>"


def build_examples(
    target_text: str,
    events: list[Event],
    labels: SessionLabels,
    mode: str,
    max_events: int = 512,
) -> list[dict]:
    """Cuts a session into windows of at most `max_events`.

    Long essays exceed the context window, so each continuation window carries
    the buffer state as it stood when that window began, and its completion
    opens with the <DT:k> spanning the window boundary -- serializing windows
    independently would silently zero that gap, which in composition can be a
    minutes-long thinking pause.
    """
    examples: list[dict] = []
    buf = TextBuffer()
    prev_press: int | None = None

    for start in range(0, len(events), max_events):
        window = events[start : start + max_events]
        prompt = (
            build_prompt(target_text, labels, mode)
            if start == 0
            else build_prompt(target_text, labels, mode, buf.text, buf.cursor)
        )
        completion = serialize(window, prev_press_time=prev_press)
        examples.append({"prompt": prompt, "completion": completion})
        prev_press = window[-1].press_time
        for e in window:
            buf.apply(e)
    return examples


def split_by_writer(
    writer_ids: Iterable[str], test_frac: float = 0.1, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Split held out by writer, never by session, so no writer leaks across.

    Each writer is assigned independently, by hashing its ID with the seed, so
    the assignment does not depend on which other writers happen to be present.
    That matters because the corpus is routinely built at several sizes
    (`--limit-aalto`): under the previous shuffle-and-take-a-prefix scheme, a
    subset build and a full build disagreed about nearly every writer, so a
    checkpoint trained on the subset could not be compared against one trained
    on the full corpus, and a writer held out from one was inside the other's
    training set.

    The cost is that the test fraction is now statistical rather than exact —
    an unavoidable consequence of deciding each writer without reference to the
    rest. At corpus scale the error is negligible (165k writers lands within a
    few parts in ten thousand of `test_frac`); on tiny inputs it is visible.

    `hashlib` is used rather than `hash()`, whose string seed is randomized per
    process, which would reshuffle the holdout on every run.
    """
    threshold = test_frac * 2**64
    train: set[str] = set()
    test: set[str] = set()
    for writer in sorted(set(writer_ids)):
        digest = hashlib.blake2b(
            f"{seed}:{writer}".encode(), digest_size=8
        ).digest()
        (test if int.from_bytes(digest, "big") < threshold else train).add(writer)
    return train, test
