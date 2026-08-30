"""Converts event streams to and from the token grammar the LLM emits.

Format v2 — full spec and the measurements behind it in docs/token-format.md.

One keystroke is two tokens, and rollover reads straight off the stream:

    <H:51><DT:49><e:51><DT:50><SPC:51>
      |      |
      |      press-to-press gap to the NEXT event
      key + its hold time, on the SAME 128-bin scale

    <X:y><DT:z> with y > z  ->  X was still held when the next key went down.

Design decisions, each measured (see the doc):
  - char and hold merge into one token: -32% sequence length for ~12.4k new
    vocabulary entries, trivial next to a 152k base vocabulary.
  - DT stays separate: merging it too would need ~1.5M entries (5.6B params).
  - holds and gaps share one bin scale so their indices compare directly;
    that is what keeps the rollover rule a plain integer comparison.
  - no leading <DT:0>: it was byte-identical in 100% of 400k examples. A
    continuation window MAY begin with <DT:k>, carrying the real gap across
    the window boundary (pass prev_press_time to serialize()).
  - condition knobs and prompt markers are single registered tokens; the v1
    text header cost 28 base-tokenizer pieces for 5 numbers.
"""

from __future__ import annotations

import re
import math
import string

from typeshi import config
from typeshi.events import Event, EventType
from typeshi.timebins import from_bin, to_bin

_ESCAPES = {" ": "SPC", "\n": "NL", "\t": "TAB", "<": "LT", ">": "GT"}
_UNESCAPES = {v: k for k, v in _ESCAPES.items()}

# Printable ASCII minus the escaped ones; these appear literally in tokens.
_DIRECT_CHARS = [c for c in string.printable[:95] if c not in _ESCAPES]

# Condition-knob binning. WPM in 5-wpm buckets (max observed in the corpus is
# 156, so 40 bins never clamp); error/revision knobs in whole percentage
# points clamped at 30% -- anything above just means "extremely high".
WPM_BIN_WIDTH = 5
WPM_BINS = 40
PCT_MAX = 30

MARKERS = ("<TARGET>", "<WRITTEN>", "<PROCESS>")

# Alternation order matters: named forms first, then the single-char form
# (whose [^<>] would otherwise swallow the first letter of DT/CUR/...).
_TOKEN_RE = re.compile(
    r"<DT:(?P<dt>\d+)>"
    r"|<CUR:(?P<cur>\d+)>"
    r"|<SELDEL:(?P<sa>\d+)-(?P<sb>\d+)>"
    r"|<BKSP:(?P<bh>\d+)>"
    r"|<(?P<ch>SPC|NL|TAB|LT|GT|[^<>]):(?P<h>\d+)>"
)


def _encode_char(c: str) -> str:
    return _ESCAPES.get(c, c)


def _decode_char(s: str) -> str:
    return _UNESCAPES.get(s, s)


def wpm_bin(wpm: float) -> int:
    return min(max(int(wpm // WPM_BIN_WIDTH), 0), WPM_BINS - 1)


def pct_bin(rate: float) -> int:
    """Rate in [0,1] -> whole percentage points, clamped at PCT_MAX."""
    return min(max(int(round(rate * 100)), 0), PCT_MAX)


# Revision rate shares <REV:>'s 31 values with the error knobs but not their
# magnitudes. ECOR genuinely spans the full range (14.8% of composition rows
# clamp at the 30% ceiling); revisions live near 1%. Measured over the 24,909
# exported composition windows, whole percents put the MEDIAN window (0.391%
# revisions) in bin 0 alongside windows that never revise, and 90.9% of all
# windows into bins 0-2 -- 31 token values expressing three states, which is
# why the knob conditions nothing.
#
# Geometric bins, the shape timebins.py already uses for timing, spread that
# mass across bins 1-30 while keeping the same 30% ceiling: the median window
# lands at 8, real writers (1.1-1.3%) at 13-14, a heavy reviser (8%) at 23.
# Bin 0 stays exactly zero, so "never revises" remains sayable. The vocabulary
# is unchanged -- only what n means moves.
REV_MIN = 0.001  # 0.1%, about one op per thousand events
REV_MAX = 0.30   # the ceiling pct_bin used, kept


def _rev_step() -> float:
    return math.log(REV_MAX / REV_MIN) / (PCT_MAX - 1)


def rev_bin(rate: float) -> int:
    """Revision rate in [0,1] -> geometric bin 0..PCT_MAX."""
    if rate <= 0:
        return 0
    clamped = min(max(rate, REV_MIN), REV_MAX)
    k = 1 + round(math.log(clamped / REV_MIN) / _rev_step())
    return min(max(k, 1), PCT_MAX)


def rev_from_bin(k: int) -> float:
    """Bin -> the revision rate it represents. Inverse of rev_bin."""
    if k <= 0:
        return 0.0
    return REV_MIN * math.exp(_rev_step() * (min(k, PCT_MAX) - 1))


def special_tokens() -> list[str]:
    """Every fixed token to register with the tokenizer.

    CUR and SELDEL carry unbounded integers, so they are emitted as plain
    text and parsed by regex rather than being single vocabulary entries;
    only their prefixes appear here, and prepare_tokenizer skips them.
    """
    toks = ["<CUR:", "<SELDEL:"]
    toks += [f"<DT:{k}>" for k in range(config.TIME_BINS)]
    chars = _DIRECT_CHARS + list(_ESCAPES.values())
    toks += [f"<{c}:{h}>" for c in chars for h in range(config.TIME_BINS)]
    toks += [f"<BKSP:{h}>" for h in range(config.TIME_BINS)]
    toks += ["<MODE:T>", "<MODE:C>"]
    toks += [f"<WPM:{b}>" for b in range(WPM_BINS)]
    for knob in ("ECOR", "EUNC", "REV"):
        toks += [f"<{knob}:{b}>" for b in range(PCT_MAX + 1)]
    toks += list(MARKERS)
    return toks


def supported_chars() -> frozenset[str]:
    """The 97 character identities that have a registered `<c:h>` token.

    The authority for "can this model type that". Callers previously read it
    off typeshi.tiny_tokenizer.TEXT_CHARS, which happens to hold the same 97
    characters but means something different: that is the tiny model's PROMPT
    vocabulary. For a Qwen checkpoint the prompt tokenizer accepts any
    Unicode happily and the constraint is entirely about which characters can
    be EMITTED, so validating against the prompt side would let a curly quote
    through to fail much later and much less legibly.
    """
    return frozenset(_DIRECT_CHARS) | frozenset(_ESCAPES)


# Word-processor homoglyphs of supported characters. Each entry is a
# RENDERING variant of an exact ASCII identity -- curly quotes, dash family,
# ellipsis, exotic spaces, zero-widths -- which is why mapping them is
# lossless. Accented letters and emoji are semantic differences and are
# deliberately absent: what to do with those belongs to the caller's error
# path, not a silent rewrite.
_HOMOGLYPHS = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
    " ": " ", " ": " ",
    **{chr(c): " " for c in range(0x2000, 0x200B)},
    "​": "", "‌": "", "‍": "", "﻿": "",
    "\r": "\n",
}


def normalize_typable(text: str) -> str:
    """Maps word-processor homoglyphs onto the 97 supported identities.

    Pasted prose almost always carries curly quotes, em dashes and CRLF;
    without this, every such paste is an error to the user and a dropped
    session to the corpus build. The output is NOT guaranteed fully
    supported -- unmappable characters pass through unchanged so the caller
    can reject them legibly.
    """
    return "".join(_HOMOGLYPHS.get(c, c) for c in text.replace("\r\n", "\n"))


def unsupported_chars(events: list[Event]) -> set[str]:
    """Characters in KEY events that have no registered <c:h> token.

    Only 97 printable-ASCII identities are in the vocabulary. Anything else
    (Cyrillic from multilingual Aalto participants, U+FFFD from KLiCKe's lossy
    decoding, curly quotes) would serialize into a syntactically valid but
    UNREGISTERED token that the tokenizer shatters into BPE pieces -- and it
    violates the plan's English-only scope. Sessions containing any are
    dropped at dataset build; measured 6 of 4,680 sample examples (0.13%).
    """
    supported = supported_chars()
    return {
        e.char for e in events
        if e.type is EventType.KEY and e.char not in supported
    }


def serialize(events: list[Event], prev_press_time: int | float | None = None) -> str:
    """Events -> token stream.

    `prev_press_time` is the press time of the event immediately before this
    window; when given, the stream opens with the <DT:k> spanning the window
    boundary instead of silently dropping that gap.
    """
    parts: list[str] = []
    prev = prev_press_time
    for e in events:
        if prev is not None:
            parts.append(f"<DT:{to_bin(e.press_time - prev)}>")
        prev = e.press_time

        if e.type is EventType.KEY:
            hold = to_bin(e.release_time - e.press_time)
            parts.append(f"<{_encode_char(e.char)}:{hold}>")
        elif e.type is EventType.BACKSPACE:
            hold = to_bin(e.release_time - e.press_time)
            parts.append(f"<BKSP:{hold}>")
        elif e.type is EventType.CURSOR:
            parts.append(f"<CUR:{e.pos}>")
        elif e.type is EventType.SELDEL:
            parts.append(f"<SELDEL:{e.start}-{e.end}>")
    return "".join(parts)


def codec_roundtrip(events: list[Event]) -> list[Event]:
    """Events -> tokens -> events: projects a session onto the codec's grid.

    Every timing in a serialized session lives on the 128-bin log grid, so a
    generated session can only ever take ~40 distinct hold values where raw
    corpus sessions take hundreds. A discriminator fed raw real sessions
    against generated ones separates them on that comb alone -- measured
    0.8275 paired CV falling to 0.6200 once the real side was roundtripped,
    with 0.438 of the GBM's importance on a single hold quantile. Comparing
    realism in-representation requires both sides on the grid; this helper is
    that projection. Idempotent: bin centers map to themselves.
    """
    return deserialize(serialize(events))


def deserialize(text: str) -> list[Event]:
    tokens = list(_TOKEN_RE.finditer(text))
    consumed = sum(len(m.group(0)) for m in tokens)
    if consumed != len(text):
        raise ValueError("input contains text outside the event grammar")

    events: list[Event] = []
    clock = 0
    pending_dt: int | None = None

    for m in tokens:
        if m.group("dt") is not None:
            if pending_dt is not None:
                raise ValueError("two <DT:> tokens in a row")
            pending_dt = from_bin(int(m.group("dt")))
            continue

        # Every event after the first must be preceded by a <DT:>. The first
        # may carry one too (a continuation window) or start at the clock.
        if events and pending_dt is None:
            raise ValueError(f"event token {m.group(0)} not preceded by <DT:>")
        clock += pending_dt or 0
        pending_dt = None

        if m.group("ch") is not None:
            hold = from_bin(int(m.group("h")))
            events.append(Event.key(_decode_char(m.group("ch")), clock, clock + hold))
        elif m.group("bh") is not None:
            hold = from_bin(int(m.group("bh")))
            events.append(Event.backspace(clock, clock + hold))
        elif m.group("cur") is not None:
            events.append(Event.cursor(int(m.group("cur")), clock))
        else:
            events.append(Event.seldel(int(m.group("sa")), int(m.group("sb")), clock))

    if pending_dt is not None:
        raise ValueError("stream ends on a dangling <DT:>")
    return events
