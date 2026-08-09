"""Canonical keystroke event representation shared by every corpus adapter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    KEY = "key"
    BACKSPACE = "bksp"
    CURSOR = "cursor"
    SELDEL = "seldel"


@dataclass(frozen=True)
class Event:
    type: EventType
    press_time: int
    release_time: int | None = None
    char: str | None = None
    pos: int | None = None
    start: int | None = None
    end: int | None = None

    @staticmethod
    def key(char: str, press_time: int, release_time: int) -> "Event":
        if len(char) != 1:
            raise ValueError(f"key event needs exactly one char, got {char!r}")
        return Event(EventType.KEY, press_time, release_time, char=char)

    @staticmethod
    def backspace(press_time: int, release_time: int) -> "Event":
        return Event(EventType.BACKSPACE, press_time, release_time)

    @staticmethod
    def cursor(pos: int, press_time: int) -> "Event":
        if pos < 0:
            raise ValueError(f"cursor position must be non-negative, got {pos}")
        return Event(EventType.CURSOR, press_time, pos=pos)

    @staticmethod
    def seldel(start: int, end: int, press_time: int) -> "Event":
        if start >= end:
            raise ValueError(f"seldel needs start < end, got {start} >= {end}")
        return Event(EventType.SELDEL, press_time, start=start, end=end)
