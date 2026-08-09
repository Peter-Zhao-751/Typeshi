"""Turns parsed sessions into prompt/completion training examples."""

from __future__ import annotations

import random
from typing import Iterable

from typeshi.buffer import TextBuffer
from typeshi.events import Event
from typeshi.labels import SessionLabels
from typeshi.serialize import serialize

_PROMPT = (
    "Simulate the writing process for the target text.\n"
    "{header}\n"
    "TARGET: {target}\n"
    "{state}"
    "PROCESS:"
)


def build_prompt(
    target_text: str,
    labels: SessionLabels,
    mode: str,
    written_so_far: str = "",
    cursor: int | None = None,
) -> str:
    """The prompt format shared by training export and inference."""
    state = ""
    if cursor is not None:
        # Resume state: how far along the writer is, and where the caret sits.
        state = f"WRITTEN_SO_FAR: {written_so_far}\nCURSOR={cursor}\n"
    return _PROMPT.format(header=labels.to_header(mode), target=target_text, state=state)


def build_examples(
    target_text: str,
    events: list[Event],
    labels: SessionLabels,
    mode: str,
    max_events: int = 512,
) -> list[dict]:
    """Cuts a session into windows of at most `max_events`.

    Long essays exceed the context window, so each continuation window carries
    the buffer state as it stood when that window began.
    """
    examples: list[dict] = []
    buf = TextBuffer()

    for start in range(0, len(events), max_events):
        window = events[start : start + max_events]
        prompt = (
            build_prompt(target_text, labels, mode)
            if start == 0
            else build_prompt(target_text, labels, mode, buf.text, buf.cursor)
        )
        examples.append({"prompt": prompt, "completion": serialize(window)})
        for e in window:
            buf.apply(e)
    return examples


def split_by_writer(
    writer_ids: Iterable[str], test_frac: float = 0.1, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Split held out by writer, never by session, so no writer leaks across."""
    ids = sorted(set(writer_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    return set(ids[n_test:]), set(ids[:n_test])
