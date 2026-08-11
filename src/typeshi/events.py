"""Canonical keystroke event representation shared by every corpus adapter."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class EventType(Enum):
    KEY = "key"
    BACKSPACE = "bksp"
    CURSOR = "cursor"
    SELDEL = "seldel"


@dataclass(frozen=True)
class Event:
    """Canonical keystroke event with timing.

    Times are integers (raw milliseconds) when parsed from adapters, but floats
    (bin-center milliseconds) when deserialized from the token format. The
    symmetrized eval uses deserialized times for all discriminator inputs.
    """
    type: EventType
    press_time: int | float
    release_time: int | float | None = None
    char: str | None = None
    pos: int | None = None
    start: int | None = None
    end: int | None = None

    @staticmethod
    def key(char: str, press_time: int | float, release_time: int | float) -> "Event":
        if len(char) != 1:
            raise ValueError(f"key event needs exactly one char, got {char!r}")
        return Event(EventType.KEY, press_time, release_time, char=char)

    @staticmethod
    def backspace(press_time: int | float, release_time: int | float) -> "Event":
        return Event(EventType.BACKSPACE, press_time, release_time)

    @staticmethod
    def cursor(pos: int, press_time: int | float) -> "Event":
        if pos < 0:
            raise ValueError(f"cursor position must be non-negative, got {pos}")
        return Event(EventType.CURSOR, press_time, pos=pos)

    @staticmethod
    def seldel(start: int, end: int, press_time: int | float) -> "Event":
        if start >= end:
            raise ValueError(f"seldel needs start < end, got {start} >= {end}")
        return Event(EventType.SELDEL, press_time, start=start, end=end)


def rebase(events: Sequence[Event]) -> list[Event]:
    """Shifts times so the first event sits at 0.

    Every adapter needs this and none can do it while parsing: logs open with
    rows that produce no event (a stray SHIFT, a click), and anchoring on the
    first *row* would leave that dead time in front of the session.
    """
    if not events:
        return list(events)
    t0 = events[0].press_time
    return [
        dataclasses.replace(
            e,
            press_time=e.press_time - t0,  # type: ignore[operator]
            release_time=None if e.release_time is None else e.release_time - t0,  # type: ignore[operator]
        )
        for e in events
    ]
