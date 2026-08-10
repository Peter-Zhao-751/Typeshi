"""Turns parsed sessions into prompt/completion training examples."""

from __future__ import annotations

import random
from typing import Iterable

from typeshi.buffer import TextBuffer
from typeshi.events import Event
from typeshi.labels import SessionLabels
from typeshi.serialize import serialize

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
    state = ""
    if cursor is not None:
        # Resume state: what stands in the buffer, and where the caret sits.
        state = f"<WRITTEN>{written_so_far}<CUR:{cursor}>"
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
    """Split held out by writer, never by session, so no writer leaks across."""
    ids = sorted(set(writer_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = int(round(len(ids) * test_frac))
    return set(ids[n_test:]), set(ids[:n_test])
