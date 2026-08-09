"""Applies event streams to a text buffer. The single source of truth for
what a sequence of keystrokes actually produces."""

from __future__ import annotations

from typing import Iterable

from typeshi.events import Event, EventType


class ReplayError(Exception):
    """An event could not be applied to the buffer."""


class TextBuffer:
    def __init__(self, text: str = "", cursor: int | None = None) -> None:
        self.text = text
        self.cursor = len(text) if cursor is None else cursor

    def apply(self, event: Event) -> None:
        if event.type is EventType.KEY:
            self._insert(event.char)
        elif event.type is EventType.BACKSPACE:
            self._backspace()
        elif event.type is EventType.CURSOR:
            self._move(event.pos)
        elif event.type is EventType.SELDEL:
            self._seldel(event.start, event.end)
        else:  # pragma: no cover - enum is exhaustive
            raise ReplayError(f"unknown event type {event.type}")

    def _insert(self, char: str) -> None:
        self.text = self.text[: self.cursor] + char + self.text[self.cursor :]
        self.cursor += 1

    def _backspace(self) -> None:
        if self.cursor == 0:
            return
        self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
        self.cursor -= 1

    def _move(self, pos: int) -> None:
        if not 0 <= pos <= len(self.text):
            raise ReplayError(f"cursor {pos} outside buffer of length {len(self.text)}")
        self.cursor = pos

    def _seldel(self, start: int, end: int) -> None:
        if not 0 <= start < end <= len(self.text):
            raise ReplayError(
                f"seldel [{start},{end}) outside buffer of length {len(self.text)}"
            )
        self.text = self.text[:start] + self.text[end:]
        self.cursor = start


def replay(events: Iterable[Event]) -> str:
    buf = TextBuffer()
    for e in events:
        buf.apply(e)
    return buf.text
