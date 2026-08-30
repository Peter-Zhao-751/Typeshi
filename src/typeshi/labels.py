"""Per-session condition labels. These become the prompt knobs, so the model
learns the knob -> behavior mapping from real variation between typists."""

from __future__ import annotations

from dataclasses import dataclass

from typeshi.buffer import TextBuffer, replay
from typeshi.events import Event, EventType


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class SessionLabels:
    wpm: float
    corrected_error_rate: float
    uncorrected_error_rate: float
    revision_rate: float

    def to_tokens(self, mode: str) -> str:
        """Condition knobs as single registered tokens (format v2).

        The v1 text header ("MODE=transcription WPM=92 ERR_COR=2.3% ...")
        shredded into 28 base-tokenizer pieces for five numbers; these five
        tokens carry the same conditioning signal.
        """
        from typeshi.serialize import pct_bin, rev_bin, wpm_bin

        m = "T" if mode == "transcription" else "C"
        return (
            f"<MODE:{m}>"
            f"<WPM:{wpm_bin(self.wpm)}>"
            f"<ECOR:{pct_bin(self.corrected_error_rate)}>"
            f"<EUNC:{pct_bin(self.uncorrected_error_rate)}>"
            f"<REV:{rev_bin(self.revision_rate)}>"
        )


def window_labels(events: list[Event], session: SessionLabels,
                  start_text: str = "",
                  start_cursor: int | None = None) -> SessionLabels:
    """Labels describing ONE window, inheriting only what is session-wide.

    Speed, correction rate and revision rate are properties of the stretch of
    typing in front of you, and attaching the whole session's values to every
    window is a mislabelling: measured over 24,909 exported composition
    windows, the <REV:> bin matched its own window only 52.9% of the time.
    39.9% of windows labelled <REV:0> do revise. A third of the signal taught
    the model that the token predicts nothing.

    `uncorrected_error_rate` is the exception and stays session-wide: it is how
    far the FINAL text lands from the target, and a window that has typed half
    the text has not made an error by stopping there.

    Takes no target_text on purpose -- the three window-local rates do not need
    it, and computing the fourth per window would run a Levenshtein against the
    full target for every window of every session.
    """
    if not events:
        return SessionLabels(0.0, 0.0, 0.0, session.uncorrected_error_rate)

    # Replayed from the buffer AS IT STOOD, not from empty: a continuation
    # window's <CUR:p> indexes the whole session's text, and replaying it
    # against an empty buffer raises ReplayError (seen in a real export:
    # "cursor 651 outside buffer of length 271").
    #
    # Text GROWN over the window, not the keystroke count: WPM is words
    # produced per minute, and a stretch of typing with corrections presses
    # more keys than it keeps. For a first window (start_text empty) this is
    # exactly len(replay(events)), so a session too short to window is
    # labelled identically either way. A window that only revises grows
    # nothing and is legitimately ~0 wpm.
    buf = TextBuffer(start_text, start_cursor)
    for event in events:
        buf.apply(event)
    grown = max(0, len(buf.text) - len(start_text))
    duration_ms = events[-1].press_time - events[0].press_time
    minutes = duration_ms / 60_000
    wpm = (grown / 5) / minutes if minutes > 0 else 0.0

    keys = sum(1 for e in events if e.type is EventType.KEY)
    deletions = sum(1 for e in events if e.type is EventType.BACKSPACE)
    deletions += sum(
        e.end - e.start for e in events if e.type is EventType.SELDEL
    )
    corrected = deletions / keys if keys else 0.0

    revisions = sum(
        1 for e in events if e.type in (EventType.CURSOR, EventType.SELDEL)
    )

    return SessionLabels(
        wpm=wpm,
        corrected_error_rate=min(corrected, 1.0),
        uncorrected_error_rate=session.uncorrected_error_rate,
        revision_rate=revisions / len(events),
    )


def compute_labels(events: list[Event], target_text: str) -> SessionLabels:
    if not events:
        return SessionLabels(0.0, 0.0, 0.0, 0.0)

    produced = replay(events)
    duration_ms = events[-1].press_time - events[0].press_time
    minutes = duration_ms / 60_000
    wpm = (len(produced) / 5) / minutes if minutes > 0 else 0.0

    keys = sum(1 for e in events if e.type is EventType.KEY)
    deletions = sum(1 for e in events if e.type is EventType.BACKSPACE)
    deletions += sum(
        e.end - e.start for e in events if e.type is EventType.SELDEL
    )
    corrected = deletions / keys if keys else 0.0

    uncorrected = (
        _levenshtein(produced, target_text) / len(target_text) if target_text else 0.0
    )

    revisions = sum(
        1 for e in events if e.type in (EventType.CURSOR, EventType.SELDEL)
    )
    revision_rate = revisions / len(events)

    return SessionLabels(
        wpm=wpm,
        corrected_error_rate=min(corrected, 1.0),
        uncorrected_error_rate=min(uncorrected, 1.0),
        revision_rate=revision_rate,
    )
