"""Turns parsed sessions into prompt/completion training examples."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from typeshi.buffer import TextBuffer
from typeshi.events import Event
from typeshi.labels import SessionLabels, window_labels
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
        # Labels describe THIS window. Passing the session's down made the
        # conditioning tokens describe something the completion does not do:
        # 39.9% of exported windows labelled <REV:0> contain cursor ops.
        wl = window_labels(window, labels, buf.text, buf.cursor)
        prompt = (
            build_prompt(target_text, wl, mode)
            if start == 0
            else build_prompt(target_text, wl, mode, buf.text, buf.cursor)
        )
        completion = serialize(window, prev_press_time=prev_press)
        examples.append({"prompt": prompt, "completion": completion})
        prev_press = window[-1].press_time
        for e in window:
            buf.apply(e)
    return examples


def window_label_schedule(
    events: list[Event],
    labels: SessionLabels,
    max_events: int = 512,
) -> list[SessionLabels]:
    """The per-window labels a real session would train under.

    generate_windowed conditions window w on entry w of a label sequence;
    handing it this schedule -- computed from a real paired session exactly
    the way build_examples labels training windows -- is what keeps
    generation-time conditioning inside the distribution the model was
    taught. The equivalence is pinned by test: any drift between this walk
    and build_examples' is a bug.
    """
    schedule: list[SessionLabels] = []
    buf = TextBuffer()
    for start in range(0, len(events), max_events):
        window = events[start : start + max_events]
        schedule.append(window_labels(window, labels, buf.text, buf.cursor))
        for e in window:
            buf.apply(e)
    return schedule


_REV_TOKEN = re.compile(r"<REV:(\d+)>")


def revision_repeats(prompt: str, factor: int, min_bin: int) -> int:
    """How many times to write a training window, to rebalance revisions.

    Measured over 24,909 exported composition windows: 87% sit at REV <= 1 and
    only 0.9% at REV >= 5. The model has effectively never seen deliberate
    revision, so unblocking it in the decoder does not summon it. Duplicating
    the rare windows is the cheapest way to put the behaviour in front of it
    without new data.

    Only meaningful AFTER labels became per-window: applied to session-labelled
    windows it would have duplicated whatever the session average happened to
    say, which matched the window only 52.9% of the time.

    Returns 1 unless asked otherwise, so the export stays byte-identical by
    default.
    """
    if factor <= 1:
        return 1
    match = _REV_TOKEN.search(prompt)
    if match is None:
        return 1
    return factor if int(match.group(1)) >= min_bin else 1


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
