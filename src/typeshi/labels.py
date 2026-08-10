"""Per-session condition labels. These become the prompt knobs, so the model
learns the knob -> behavior mapping from real variation between typists."""

from __future__ import annotations

from dataclasses import dataclass

from typeshi.buffer import replay
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
        from typeshi.serialize import pct_bin, wpm_bin

        m = "T" if mode == "transcription" else "C"
        return (
            f"<MODE:{m}>"
            f"<WPM:{wpm_bin(self.wpm)}>"
            f"<ECOR:{pct_bin(self.corrected_error_rate)}>"
            f"<EUNC:{pct_bin(self.uncorrected_error_rate)}>"
            f"<REV:{pct_bin(self.revision_rate)}>"
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
