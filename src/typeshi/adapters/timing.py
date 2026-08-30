"""Keystroke timing pools harvested from real sessions.

IteraTeR gives edit STRUCTURE with no timing at all -- the notes call it
"structure ground truth, not behavior ground truth" -- so synthesis has to
put times on the keys, and inventing them from constants would teach the
model invented timing. These pools are drawn from real composition sessions
(KLiCKe) using the same boundary classification the eval's pause-position
signatures use, so a synthesized stream's timing marginals are real ones:
hold/gap pairs conditioned on clause/word/within-word position, think
pauses read off the gaps that precede real cursor/seldel traffic, and
motor gaps between consecutive ops.

Limitation, stated rather than hidden: pairs are drawn independently per
key, so within-burst serial autocorrelation is not reproduced and pair
content is uncorrelated with the specific bigram. Synthesized rows are
training data mixed in with real KLiCKe windows, never eval subjects.
"""

from __future__ import annotations

import random
from pathlib import Path

from typeshi.buffer import TextBuffer
from typeshi.events import Event, EventType

PAUSE_CLASSES = ("clause_boundary", "word_boundary", "within_word")

_CLAUSE_CHARS = frozenset(".,;!?")

_OPS = (EventType.CURSOR, EventType.SELDEL)


def pause_class(before_caret: str) -> str:
    """The eval's three-way split (signatures.pause_at_boundaries), applied
    to the text before the caret at the moment a key goes down."""
    if (len(before_caret) >= 2 and before_caret[-1] == " "
            and before_caret[-2] in _CLAUSE_CHARS):
        return "clause_boundary"
    if before_caret.endswith(" "):
        return "word_boundary"
    return "within_word"


class TimingSampler:
    """Draws (gap, hold) pairs, think pauses and op gaps from real pools."""

    def __init__(
        self,
        key_pairs: dict[str, list[tuple[float, float]]],
        think_pauses: list[float],
        op_gaps: list[float],
        seed: int = 0,
    ) -> None:
        if not key_pairs.get("within_word"):
            raise ValueError("no within-word timing pairs harvested -- the "
                             "source sessions carry no plain typing")
        # Sparse-pool fallbacks degrade toward the nearest class rather than
        # invented constants: a missing clause pool borrows word-boundary
        # pauses, a missing think pool borrows the slowest key gaps.
        self.key_pairs = {
            "within_word": list(key_pairs["within_word"]),
            "word_boundary": list(key_pairs.get("word_boundary")
                                  or key_pairs["within_word"]),
        }
        self.key_pairs["clause_boundary"] = list(
            key_pairs.get("clause_boundary")
            or self.key_pairs["word_boundary"]
        )
        slowest = sorted(g for g, _ in self.key_pairs["clause_boundary"])
        self.think_pauses = list(think_pauses) or slowest[-3:]
        self.op_gaps = list(op_gaps) or [g for g, _ in
                                         self.key_pairs["within_word"]]
        self._rng = random.Random(seed)

    def key_timing(self, cls: str) -> tuple[float, float]:
        return self._rng.choice(self.key_pairs[cls])

    def think_pause(self) -> float:
        return self._rng.choice(self.think_pauses)

    def op_gap(self) -> float:
        return self._rng.choice(self.op_gaps)

    @classmethod
    def from_sessions(cls, sessions, seed: int = 0) -> "TimingSampler":
        key_pairs: dict[str, list[tuple[float, float]]] = {
            c: [] for c in PAUSE_CLASSES
        }
        think_pauses: list[float] = []
        op_gaps: list[float] = []
        for events in sessions:
            buf = TextBuffer()
            prev: Event | None = None
            for e in events:
                if prev is not None:
                    gap = float(e.press_time - prev.press_time)
                    if gap > 0:
                        if (e.type is EventType.KEY
                                and e.release_time is not None):
                            hold = float(e.release_time - e.press_time)
                            if hold > 0:
                                c = pause_class(buf.text[: buf.cursor])
                                key_pairs[c].append((gap, hold))
                        elif e.type in _OPS:
                            if prev.type in _OPS:
                                op_gaps.append(gap)
                            else:
                                think_pauses.append(gap)
                buf.apply(e)
                prev = e
        return cls(key_pairs, think_pauses, op_gaps, seed=seed)

    @classmethod
    def from_klicke(cls, root: Path, limit_files: int = 40,
                    seed: int = 0) -> "TimingSampler":
        """Pools from the first `limit_files` KLiCKe logs (sorted, so the
        harvest is deterministic). 40 files is thousands of pairs."""
        from typeshi.adapters import klicke

        files = [f for f in sorted(Path(root).rglob("*.csv"))
                 if klicke.gold_text_path(f) is not None][:limit_files]
        sessions = (events for f in files
                    for _, _, events in klicke.iter_sessions(f))
        return cls.from_sessions(sessions, seed=seed)
