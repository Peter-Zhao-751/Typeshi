"""Converts event streams to and from the token grammar the LLM emits.

Grammar (one event): <DT:k> then the event token, plus <HOLD:k> for keys.
DT is press-to-press so it is never negative; rollover shows up as HOLD > DT.
"""

from __future__ import annotations

import re
import string

from typeshi import config
from typeshi.events import Event, EventType
from typeshi.timebins import from_bin, to_bin

_ESCAPES = {" ": "SPC", "\n": "NL", "\t": "TAB", "<": "LT", ">": "GT"}
_UNESCAPES = {v: k for k, v in _ESCAPES.items()}

# Printable ASCII minus the escaped ones; these are typed directly.
_DIRECT_CHARS = [c for c in string.printable[:95] if c not in _ESCAPES]

_TOKEN_RE = re.compile(r"<(DT|HOLD|KEY|CUR|SELDEL|BKSP)(?::([^>]*))?>")


def _encode_char(c: str) -> str:
    return _ESCAPES.get(c, c)


def _decode_char(s: str) -> str:
    return _UNESCAPES.get(s, s)


def special_tokens() -> list[str]:
    """Every fixed token to register with the tokenizer.

    CUR and SELDEL carry unbounded integers, so they are emitted as plain text
    and parsed by regex rather than being single vocabulary entries.
    """
    toks = ["<BKSP>", "<CUR:", "<SELDEL:"]
    toks += [f"<DT:{k}>" for k in range(config.TIME_BINS)]
    toks += [f"<HOLD:{k}>" for k in range(config.TIME_BINS)]
    toks += [f"<KEY:{_encode_char(c)}>" for c in _DIRECT_CHARS]
    toks += [f"<KEY:{name}>" for name in _ESCAPES.values()]
    return toks


def serialize(events: list[Event]) -> str:
    parts: list[str] = []
    prev_press = events[0].press_time if events else 0
    for i, e in enumerate(events):
        dt = 0 if i == 0 else e.press_time - prev_press
        parts.append(f"<DT:{to_bin(dt)}>")
        prev_press = e.press_time

        if e.type is EventType.KEY:
            parts.append(f"<KEY:{_encode_char(e.char)}>")
        elif e.type is EventType.BACKSPACE:
            parts.append("<BKSP>")
        elif e.type is EventType.CURSOR:
            parts.append(f"<CUR:{e.pos}>")
        elif e.type is EventType.SELDEL:
            parts.append(f"<SELDEL:{e.start}-{e.end}>")

        if e.release_time is not None:
            parts.append(f"<HOLD:{to_bin(e.release_time - e.press_time)}>")
    return "".join(parts)


def deserialize(text: str) -> list[Event]:
    tokens = list(_TOKEN_RE.finditer(text))
    consumed = sum(len(m.group(0)) for m in tokens)
    if consumed != len(text):
        raise ValueError("input contains text outside the event grammar")

    events: list[Event] = []
    clock = 0
    pending_dt: int | None = None
    i = 0
    while i < len(tokens):
        kind, arg = tokens[i].group(1), tokens[i].group(2)
        if kind == "DT":
            pending_dt = from_bin(int(arg))
            i += 1
            continue
        if pending_dt is None:
            raise ValueError(f"event token {tokens[i].group(0)} not preceded by <DT:>")
        clock += pending_dt
        pending_dt = None

        # A trailing <HOLD:k> belongs to this event if present.
        hold = None
        if i + 1 < len(tokens) and tokens[i + 1].group(1) == "HOLD":
            hold = from_bin(int(tokens[i + 1].group(2)))

        if kind == "KEY":
            events.append(Event.key(_decode_char(arg), clock, clock + (hold or 0)))
        elif kind == "BKSP":
            events.append(Event.backspace(clock, clock + (hold or 0)))
        elif kind == "CUR":
            events.append(Event.cursor(int(arg), clock))
        elif kind == "SELDEL":
            start, end = arg.split("-")
            events.append(Event.seldel(int(start), int(end), clock))
        else:
            raise ValueError(f"unexpected token {tokens[i].group(0)}")

        i += 2 if hold is not None else 1
    return events
