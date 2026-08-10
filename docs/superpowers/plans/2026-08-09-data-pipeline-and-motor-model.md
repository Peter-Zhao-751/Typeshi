# Typing Process Model — Data Pipeline & Motor Model Implementation Plan

> **Format note (2026-08-09, post-completion):** the token grammar shown in the
> code samples below is v1, which was redesigned after all 12 tasks completed
> and before any training run. The implemented grammar is v2 — see
> `docs/token-format.md`. The task structure and everything else here still
> describes the codebase accurately.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the unified keystroke-event data pipeline and the Phase-1 motor LoRA fine-tune, reaching two milestones from the spec: byte-exact round-trip replay of real sessions, and Tier-1 (transcription) timing realism verified by distributional metrics and a learned discriminator.

**Architecture:** Raw corpora (Aalto transcription, KLiCKe composition) are parsed by per-corpus adapters into one canonical `Event` stream. A replay engine applies events to a text buffer, which gives us a round-trip correctness check. A serializer converts event streams to/from LLM token strings using log-spaced time bins. Serialized sessions plus per-session condition labels become JSONL training examples, which LoRA-fine-tune an 8B open-weight base model. An eval harness scores generations on distributional metrics and a learned real-vs-fake discriminator.

**Tech Stack:** Python 3.11+, PyTorch, HuggingFace `transformers` / `peft` / `trl` / `datasets`, `polars` for parsing, `numpy`/`scipy` for metrics, `pytest` for tests, `uv` for env management.

## Global Constraints

- **English only.** No multilingual handling anywhere in the pipeline.
- **Desktop/physical keyboard only.** Mobile Aalto data is out of scope.
- **Time unit is integer milliseconds** everywhere internally. Never floats for timestamps.
- **Time bins:** 128 log-spaced bins, `min_ms=1`, `max_ms=120000`. Same constants for `DT` and `HOLD`.
- **`DT` is press-to-press delta** (always ≥ 0), **`HOLD` is press-to-release of one key.** Rollover is represented implicitly when `HOLD > DT`. Never store negative inter-key intervals.
- **Holdout splits are by writer/participant, never by session.**
- Base model: an 8B-class open-weight instruct model (Llama or Qwen). Pin the exact ID in `config.py`; do not hardcode it in scripts.
- All randomness takes an explicit seed parameter. Default seed `0`.
- Every module gets a matching `tests/` file. Tests must not require network access or the real corpora — use fixtures.

---

## File Structure

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Deps, tool config |
| `src/typeshi/config.py` | Constants: bin params, model ID, paths |
| `src/typeshi/events.py` | Canonical `Event` / `EventType` types |
| `src/typeshi/buffer.py` | `TextBuffer` replay engine |
| `src/typeshi/timebins.py` | Log-spaced time binning |
| `src/typeshi/serialize.py` | Events ↔ token strings |
| `src/typeshi/labels.py` | Per-session condition labels (WPM, error rates) |
| `src/typeshi/adapters/aalto.py` | Aalto CSV → events |
| `src/typeshi/adapters/klicke.py` | KLiCKe CSV → events |
| `src/typeshi/dataset.py` | Windowing + JSONL export |
| `src/typeshi/train_motor.py` | Phase-1 LoRA training |
| `src/typeshi/generate.py` | Sampling harness |
| `src/typeshi/eval/distributional.py` | KL / Fréchet on timing distributions |
| `src/typeshi/eval/discriminator.py` | Real-vs-fake classifier |
| `tests/` | Mirrors `src/` layout |

---

### Task 1: Project scaffolding and canonical event types

**Files:**
- Create: `pyproject.toml`, `src/typeshi/__init__.py`, `src/typeshi/config.py`, `src/typeshi/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces: `EventType` enum with members `KEY`, `BACKSPACE`, `CURSOR`, `SELDEL`; frozen dataclass `Event(type, press_time, release_time, char, pos, start, end)` with `Event.key(char, press_time, release_time)`, `Event.backspace(press_time, release_time)`, `Event.cursor(pos, press_time)`, `Event.seldel(start, end, press_time)` constructors; `config.TIME_BINS=128`, `config.MIN_MS=1`, `config.MAX_MS=120000`, `config.BASE_MODEL`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events.py
import pytest
from typeshi.events import Event, EventType


def test_key_event_carries_char_and_times():
    e = Event.key("a", press_time=100, release_time=180)
    assert e.type is EventType.KEY
    assert e.char == "a"
    assert e.press_time == 100
    assert e.release_time == 180
    assert e.pos is None


def test_cursor_event_has_position_and_no_release():
    e = Event.cursor(pos=42, press_time=500)
    assert e.type is EventType.CURSOR
    assert e.pos == 42
    assert e.release_time is None


def test_seldel_event_has_range():
    e = Event.seldel(start=3, end=9, press_time=700)
    assert e.type is EventType.SELDEL
    assert (e.start, e.end) == (3, 9)


def test_events_are_frozen():
    e = Event.backspace(press_time=10, release_time=20)
    with pytest.raises(Exception):
        e.press_time = 99


def test_key_event_rejects_multichar():
    with pytest.raises(ValueError):
        Event.key("ab", press_time=1, release_time=2)


def test_seldel_rejects_inverted_range():
    with pytest.raises(ValueError):
        Event.seldel(start=9, end=3, press_time=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi'`

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "typeshi"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["polars>=1.0", "numpy>=1.26", "scipy>=1.11"]

[project.optional-dependencies]
train = ["torch>=2.4", "transformers>=4.44", "peft>=0.12", "trl>=0.9", "datasets>=2.20", "accelerate>=0.33"]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/typeshi"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

```python
# src/typeshi/config.py
"""Project-wide constants. Import these; never inline the values."""

TIME_BINS = 128
MIN_MS = 1
MAX_MS = 120_000

BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

DEFAULT_SEED = 0
```

```python
# src/typeshi/events.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_events.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/typeshi tests/test_events.py
git commit -m "feat: canonical keystroke event types and project scaffolding"
```

---

### Task 2: Text buffer replay engine

**Files:**
- Create: `src/typeshi/buffer.py`
- Test: `tests/test_buffer.py`

**Interfaces:**
- Consumes: `Event`, `EventType` from Task 1
- Produces: `TextBuffer` class with `.text: str`, `.cursor: int`, `.apply(event: Event) -> None`, and module function `replay(events: Iterable[Event]) -> str` returning final text. Raises `ReplayError` on invalid operations.

This is the correctness backbone: it converts any event stream into text, which is how we verify parsers and, later, how constrained decoding tracks state.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_buffer.py
import pytest
from typeshi.buffer import TextBuffer, ReplayError, replay
from typeshi.events import Event


def _type(text, t0=0):
    """Helper: build KEY events typing `text` left to right."""
    return [Event.key(c, press_time=t0 + i * 100, release_time=t0 + i * 100 + 50)
            for i, c in enumerate(text)]


def test_typing_appends_text_and_advances_cursor():
    b = TextBuffer()
    for e in _type("hello"):
        b.apply(e)
    assert b.text == "hello"
    assert b.cursor == 5


def test_backspace_deletes_char_before_cursor():
    b = TextBuffer()
    for e in _type("hello"):
        b.apply(e)
    b.apply(Event.backspace(press_time=600, release_time=650))
    assert b.text == "hell"
    assert b.cursor == 4


def test_backspace_at_start_is_a_noop():
    b = TextBuffer()
    b.apply(Event.backspace(press_time=0, release_time=10))
    assert b.text == ""
    assert b.cursor == 0


def test_cursor_move_then_insert_writes_mid_string():
    b = TextBuffer()
    for e in _type("helo"):
        b.apply(e)
    b.apply(Event.cursor(pos=3, press_time=500))
    b.apply(Event.key("l", press_time=600, release_time=650))
    assert b.text == "hello"
    assert b.cursor == 4


def test_seldel_removes_range_and_places_cursor_at_start():
    b = TextBuffer()
    for e in _type("hello world"):
        b.apply(e)
    b.apply(Event.seldel(start=5, end=11, press_time=2000))
    assert b.text == "hello"
    assert b.cursor == 5


def test_cursor_beyond_end_raises():
    b = TextBuffer()
    for e in _type("hi"):
        b.apply(e)
    with pytest.raises(ReplayError):
        b.apply(Event.cursor(pos=99, press_time=500))


def test_seldel_out_of_bounds_raises():
    b = TextBuffer()
    for e in _type("hi"):
        b.apply(e)
    with pytest.raises(ReplayError):
        b.apply(Event.seldel(start=1, end=50, press_time=500))


def test_replay_helper_returns_final_text():
    assert replay(_type("abc")) == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_buffer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.buffer'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/typeshi/buffer.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_buffer.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/buffer.py tests/test_buffer.py
git commit -m "feat: text buffer replay engine"
```

---

### Task 3: Log-spaced time binning

**Files:**
- Create: `src/typeshi/timebins.py`
- Test: `tests/test_timebins.py`

**Interfaces:**
- Consumes: `config.TIME_BINS`, `config.MIN_MS`, `config.MAX_MS`
- Produces: `bin_edges() -> np.ndarray` (length `TIME_BINS + 1`), `to_bin(dt_ms: int) -> int` (clamps out-of-range), `from_bin(k: int) -> int` (geometric midpoint, integer ms).

Log spacing gives millisecond resolution on fast keystrokes and second-scale resolution on thinking pauses, which is exactly the dynamic range typing spans.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_timebins.py
import numpy as np
import pytest
from typeshi import config
from typeshi.timebins import bin_edges, to_bin, from_bin


def test_edges_span_configured_range():
    e = bin_edges()
    assert len(e) == config.TIME_BINS + 1
    assert e[0] == pytest.approx(config.MIN_MS)
    assert e[-1] == pytest.approx(config.MAX_MS)


def test_edges_are_monotonic():
    e = bin_edges()
    assert np.all(np.diff(e) > 0)


def test_bins_are_log_spaced_not_linear():
    e = bin_edges()
    first_width = e[1] - e[0]
    last_width = e[-1] - e[-2]
    assert last_width > first_width * 100


def test_to_bin_is_in_range():
    for dt in [0, 1, 50, 200, 5_000, 500_000]:
        assert 0 <= to_bin(dt) < config.TIME_BINS


def test_to_bin_clamps_extremes():
    assert to_bin(-5) == 0
    assert to_bin(10_000_000) == config.TIME_BINS - 1


def test_to_bin_is_monotonic_nondecreasing():
    bins = [to_bin(dt) for dt in range(1, 5000, 7)]
    assert bins == sorted(bins)


def test_round_trip_error_is_bounded():
    """A value recovered from its bin should be within ~15% of the original."""
    for dt in [5, 40, 120, 900, 4_000, 30_000]:
        recovered = from_bin(to_bin(dt))
        assert abs(recovered - dt) / dt < 0.15


def test_from_bin_returns_int():
    assert isinstance(from_bin(10), int)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_timebins.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.timebins'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/typeshi/timebins.py
"""Log-spaced quantization of inter-event times.

Typing spans three orders of magnitude (2 ms rollover to 60 s thinking pause),
so bins are geometric: fine where keystrokes are fast, coarse where pauses are long.
"""

from __future__ import annotations

import functools

import numpy as np

from typeshi import config


@functools.lru_cache(maxsize=1)
def bin_edges() -> np.ndarray:
    return np.geomspace(config.MIN_MS, config.MAX_MS, config.TIME_BINS + 1)


def to_bin(dt_ms: int) -> int:
    edges = bin_edges()
    dt = min(max(float(dt_ms), config.MIN_MS), config.MAX_MS)
    # searchsorted returns the insertion index; subtract 1 for the containing bin.
    k = int(np.searchsorted(edges, dt, side="right")) - 1
    return min(max(k, 0), config.TIME_BINS - 1)


def from_bin(k: int) -> int:
    if not 0 <= k < config.TIME_BINS:
        raise ValueError(f"bin index {k} out of range")
    edges = bin_edges()
    return int(round(float(np.sqrt(edges[k] * edges[k + 1]))))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_timebins.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/timebins.py tests/test_timebins.py
git commit -m "feat: log-spaced time binning"
```

---

### Task 4: Event stream serializer (events ↔ tokens)

**Files:**
- Create: `src/typeshi/serialize.py`
- Test: `tests/test_serialize.py`

**Interfaces:**
- Consumes: `Event`, `EventType`, `to_bin`, `from_bin`
- Produces: `serialize(events: list[Event]) -> str`, `deserialize(text: str) -> list[Event]`, `special_tokens() -> list[str]` (the full list to add to the tokenizer).

Token grammar, one event = a `<DT:k>` then the event token then (for keys) `<HOLD:k>`:

```
<DT:12><KEY:h><HOLD:7><DT:9><KEY:i><HOLD:6><DT:44><BKSP><HOLD:5><DT:60><CUR:3><DT:11><SELDEL:3-9>
```

`DT` for the first event is measured from session start (i.e. `press_time` of event 0). Characters are escaped so `<KEY:>` is unambiguous: space is `<KEY:SPC>`, newline `<KEY:NL>`, `<` is `<KEY:LT>`, `>` is `<KEY:GT>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serialize.py
import pytest
from typeshi.events import Event, EventType
from typeshi.serialize import serialize, deserialize, special_tokens


def test_serialize_produces_dt_key_hold_triples():
    events = [Event.key("h", 100, 160), Event.key("i", 220, 270)]
    s = serialize(events)
    assert s.count("<KEY:h>") == 1
    assert s.count("<KEY:i>") == 1
    assert s.count("<DT:") == 2
    assert s.count("<HOLD:") == 2


def test_round_trip_preserves_event_types_and_chars():
    events = [
        Event.key("a", 100, 150),
        Event.backspace(300, 340),
        Event.cursor(0, 900),
        Event.key("b", 1000, 1050),
        Event.seldel(0, 1, 2000),
    ]
    out = deserialize(serialize(events))
    assert [e.type for e in out] == [e.type for e in events]
    assert [e.char for e in out] == [e.char for e in events]
    assert out[2].pos == 0
    assert (out[4].start, out[4].end) == (0, 1)


def test_round_trip_timing_is_close_within_bin_error():
    events = [Event.key("a", 100, 150), Event.key("b", 400, 460)]
    out = deserialize(serialize(events))
    gap_in = events[1].press_time - events[0].press_time
    gap_out = out[1].press_time - out[0].press_time
    assert abs(gap_out - gap_in) / gap_in < 0.15


def test_special_characters_are_escaped():
    events = [Event.key(" ", 0, 50), Event.key("\n", 100, 150),
              Event.key("<", 200, 250), Event.key(">", 300, 350)]
    s = serialize(events)
    assert "<KEY:SPC>" in s and "<KEY:NL>" in s
    assert "<KEY:LT>" in s and "<KEY:GT>" in s
    out = deserialize(s)
    assert [e.char for e in out] == [" ", "\n", "<", ">"]


def test_rollover_is_representable():
    """Next key pressed before previous released -> HOLD exceeds DT."""
    events = [Event.key("t", 0, 120), Event.key("h", 60, 180)]
    out = deserialize(serialize(events))
    hold0 = out[0].release_time - out[0].press_time
    dt = out[1].press_time - out[0].press_time
    assert hold0 > dt


def test_deserialize_rejects_malformed_input():
    with pytest.raises(ValueError):
        deserialize("<KEY:a><NONSENSE:1>")


def test_special_tokens_cover_grammar():
    toks = special_tokens()
    assert "<BKSP>" in toks
    assert "<DT:0>" in toks and "<DT:127>" in toks
    assert "<KEY:SPC>" in toks
    assert any(t.startswith("<KEY:a") for t in toks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_serialize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.serialize'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/typeshi/serialize.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_serialize.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/serialize.py tests/test_serialize.py
git commit -m "feat: event stream token serializer"
```

---

### Task 5: Session condition labels

**Files:**
- Create: `src/typeshi/labels.py`
- Test: `tests/test_labels.py`

**Interfaces:**
- Consumes: `Event`, `EventType`, `replay`
- Produces: frozen dataclass `SessionLabels(wpm: float, corrected_error_rate: float, uncorrected_error_rate: float, revision_rate: float)`; `compute_labels(events: list[Event], target_text: str) -> SessionLabels`; `SessionLabels.to_header(mode: str) -> str` producing the prompt header string.

WPM uses the standard convention: (characters / 5) / minutes. `corrected_error_rate` is the fraction of key events that were later deleted; `uncorrected_error_rate` is the normalized edit distance between the produced text and the intended target; `revision_rate` is the fraction of events that are `CURSOR`/`SELDEL` (non-linear editing).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labels.py
import pytest
from typeshi.events import Event
from typeshi.labels import SessionLabels, compute_labels


def _type(text, start=0, gap=100, hold=50):
    return [Event.key(c, start + i * gap, start + i * gap + hold)
            for i, c in enumerate(text)]


def test_wpm_matches_standard_formula():
    # 25 chars at exactly 100 ms/char = 2.5 s -> (25/5) / (2.5/60) = 120 wpm
    events = _type("a" * 25, gap=100)
    labels = compute_labels(events, "a" * 25)
    assert labels.wpm == pytest.approx(120, rel=0.05)


def test_clean_session_has_zero_error_rates():
    events = _type("hello")
    labels = compute_labels(events, "hello")
    assert labels.corrected_error_rate == 0.0
    assert labels.uncorrected_error_rate == 0.0


def test_typo_then_backspace_counts_as_corrected_error():
    events = _type("helz") + [Event.backspace(400, 450)] + _type("lo", start=500)
    labels = compute_labels(events, "hello")
    assert labels.corrected_error_rate > 0
    assert labels.uncorrected_error_rate == 0.0


def test_uncorrected_typo_shows_in_uncorrected_rate():
    events = _type("helzo")
    labels = compute_labels(events, "hello")
    assert labels.uncorrected_error_rate > 0
    assert labels.corrected_error_rate == 0.0


def test_revision_rate_counts_cursor_and_seldel():
    events = _type("hello") + [Event.cursor(0, 600), Event.seldel(0, 1, 700)]
    labels = compute_labels(events, "ello")
    assert labels.revision_rate == pytest.approx(2 / 7, rel=0.01)


def test_header_is_stable_and_contains_all_knobs():
    labels = SessionLabels(wpm=61.4, corrected_error_rate=0.031,
                           uncorrected_error_rate=0.004, revision_rate=0.12)
    header = labels.to_header(mode="transcription")
    assert "MODE=transcription" in header
    assert "WPM=61" in header
    assert "ERR_COR=3.1%" in header
    assert "ERR_UNC=0.4%" in header
    assert "REV=12%" in header


def test_empty_session_does_not_divide_by_zero():
    labels = compute_labels([], "")
    assert labels.wpm == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/typeshi/labels.py
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

    def to_header(self, mode: str) -> str:
        return (
            f"MODE={mode} "
            f"WPM={self.wpm:.0f} "
            f"ERR_COR={self.corrected_error_rate * 100:.1f}% "
            f"ERR_UNC={self.uncorrected_error_rate * 100:.1f}% "
            f"REV={self.revision_rate * 100:.0f}%"
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_labels.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/labels.py tests/test_labels.py
git commit -m "feat: per-session condition labels and prompt header"
```

---

### Task 6: Corpus acquisition and schema discovery

**Files:**
- Create: `scripts/fetch_data.py`, `docs/data-schemas.md`
- Test: none (this task produces documentation and downloaded bytes, verified by inspection)

**Interfaces:**
- Consumes: nothing
- Produces: raw corpora under `data/raw/aalto/` and `data/raw/klicke/`, plus `docs/data-schemas.md` recording the **actual observed** column names and dtypes for both corpora. Tasks 7 and 8 read that document.

The published papers do not pin down the CSV headers precisely enough to code against blind, so this task establishes ground truth before any parser is written. That ordering is deliberate: do not write adapters against guessed column names.

- [ ] **Step 1: Download the corpora**

Aalto 136M Keystrokes is distributed from `https://userinterfaces.aalto.fi/136Mkeystrokes/`. KLiCKe is distributed via the Google Drive link in `https://github.com/terryyutian/KLiCKe-Corpus`.

```bash
mkdir -p data/raw/aalto data/raw/klicke
# Follow each site's download instructions; both require accepting research-use terms.
# Place the extracted files directly under the directories above.
```

- [ ] **Step 2: Verify the licenses permit this use**

Read the terms shipped with each corpus. Record in `docs/data-schemas.md` under a "License" heading what each permits. **Stop and raise with the user if either forbids model training.**

- [ ] **Step 3: Write the schema inspection script**

```python
# scripts/fetch_data.py
"""Inspects downloaded corpora and prints their real schemas.

Run this before writing any adapter. Paste the output into docs/data-schemas.md.
"""

import sys
from pathlib import Path

import polars as pl


def describe(path: Path, n: int = 3) -> None:
    print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
    try:
        df = pl.read_csv(path, n_rows=1000, infer_schema_length=1000,
                         separator="\t" if path.suffix == ".tsv" else ",",
                         ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  could not parse: {exc}")
        return
    print(f"columns ({len(df.columns)}):")
    for name, dtype in zip(df.columns, df.dtypes):
        print(f"  {name:<28} {dtype}")
    print(f"\nfirst {n} rows:")
    print(df.head(n))


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    files = sorted(p for p in root.rglob("*") if p.suffix in {".csv", ".tsv", ".txt"})
    if not files:
        print(f"no CSV/TSV files under {root}")
        return
    for path in files[:10]:
        describe(path)
    print(f"\n{len(files)} data files found under {root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it and record the results**

Run: `uv run python scripts/fetch_data.py data/raw`

Create `docs/data-schemas.md` with, for each corpus: file layout (how many files, what one file contains), every column name and dtype, what one row represents, the units and epoch of timestamps, and how key identity is encoded (literal character vs. keycode). Include a "Quirks" section for anything surprising — duplicate rows, missing release times, non-monotonic timestamps, sentinel values.

- [ ] **Step 5: Commit**

```bash
git add scripts/fetch_data.py docs/data-schemas.md
git commit -m "docs: record observed corpus schemas"
```

Note: `data/` must be gitignored. Add `data/` to `.gitignore` in this commit.

---

### Task 7: Aalto adapter

**Files:**
- Create: `src/typeshi/adapters/__init__.py`, `src/typeshi/adapters/aalto.py`, `tests/fixtures/aalto_sample.csv`
- Test: `tests/test_aalto_adapter.py`

**Interfaces:**
- Consumes: `Event`, `SessionLabels`, `compute_labels`; the schema recorded in `docs/data-schemas.md`
- Produces: `AALTO_COLUMNS: dict[str, str]` mapping canonical names (`participant`, `session`, `target`, `press_ms`, `release_ms`, `char`) to the real column names from Task 6; `parse_session(rows: pl.DataFrame) -> list[Event]`; `iter_sessions(path: Path) -> Iterator[tuple[str, str, list[Event]]]` yielding `(participant_id, target_text, events)`.

Aalto is transcription: linear typing with backspaces, no cursor jumps. Only `KEY` and `BACKSPACE` events are produced.

- [ ] **Step 1: Write the fixture and failing test**

Create `tests/fixtures/aalto_sample.csv` by copying ~40 real rows spanning two participants out of the downloaded corpus, keeping the real header line verbatim. This keeps tests offline while binding them to the true schema.

```python
# tests/test_aalto_adapter.py
from pathlib import Path

import polars as pl
import pytest

from typeshi.adapters.aalto import AALTO_COLUMNS, iter_sessions, parse_session
from typeshi.buffer import replay
from typeshi.events import EventType

FIXTURE = Path(__file__).parent / "fixtures" / "aalto_sample.csv"


def test_fixture_has_the_columns_the_adapter_expects():
    """Fails loudly with the real header if the corpus schema differs."""
    df = pl.read_csv(FIXTURE, n_rows=1)
    missing = [c for c in AALTO_COLUMNS.values() if c not in df.columns]
    assert not missing, f"missing {missing}; actual columns are {df.columns}"


def test_parses_events_in_press_time_order():
    _, _, events = next(iter_sessions(FIXTURE))
    times = [e.press_time for e in events]
    assert times == sorted(times)


def test_only_key_and_backspace_events_are_emitted():
    _, _, events = next(iter_sessions(FIXTURE))
    assert {e.type for e in events} <= {EventType.KEY, EventType.BACKSPACE}


def test_press_times_are_zero_based_ms_ints():
    _, _, events = next(iter_sessions(FIXTURE))
    assert events[0].press_time == 0
    assert all(isinstance(e.press_time, int) for e in events)


def test_hold_times_are_non_negative():
    _, _, events = next(iter_sessions(FIXTURE))
    assert all(e.release_time >= e.press_time for e in events)


def test_replayed_text_is_close_to_the_recorded_target():
    """Transcription is imperfect, so allow small divergence but not garbage."""
    _, target, events = next(iter_sessions(FIXTURE))
    produced = replay(events)
    assert len(produced) > 0
    overlap = sum(a == b for a, b in zip(produced, target))
    assert overlap / max(len(target), 1) > 0.5


def test_multiple_sessions_are_yielded():
    assert len(list(iter_sessions(FIXTURE))) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aalto_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.adapters'`

- [ ] **Step 3: Write the adapter**

Fill `AALTO_COLUMNS` with the real names recorded in `docs/data-schemas.md`. The values below are the expected published names — **the first test will fail with the true header if they are wrong, and that failure is the signal to correct this dict.**

```python
# src/typeshi/adapters/aalto.py
"""Aalto 136M Keystrokes -> canonical events.

Transcription only: linear typing plus backspaces. No cursor movement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl

from typeshi.events import Event

# Canonical name -> real CSV column. Verified against docs/data-schemas.md.
AALTO_COLUMNS = {
    "participant": "PARTICIPANT_ID",
    "session": "TEST_SECTION_ID",
    "target": "SENTENCE",
    "press_ms": "PRESS_TIME",
    "release_ms": "RELEASE_TIME",
    "char": "LETTER",
}

_BACKSPACE_MARKERS = {"BKSP", "BACKSPACE", "\x08"}


def parse_session(rows: pl.DataFrame) -> list[Event]:
    c = AALTO_COLUMNS
    rows = rows.sort(c["press_ms"])
    if rows.is_empty():
        return []
    t0 = int(rows[c["press_ms"]][0])

    events: list[Event] = []
    for row in rows.iter_rows(named=True):
        press = int(row[c["press_ms"]]) - t0
        release = int(row[c["release_ms"]]) - t0
        release = max(release, press)  # guard against logging jitter
        letter = row[c["char"]]
        if letter is None:
            continue
        if str(letter).upper() in _BACKSPACE_MARKERS:
            events.append(Event.backspace(press, release))
        elif len(str(letter)) == 1:
            events.append(Event.key(str(letter), press, release))
        # Multi-char names (SHIFT, ENTER-as-word, etc.) are modifier rows: skip.
    return events


def iter_sessions(path: Path) -> Iterator[tuple[str, str, list[Event]]]:
    c = AALTO_COLUMNS
    df = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)
    for (participant, _session), group in df.group_by(
        [c["participant"], c["session"]], maintain_order=True
    ):
        target = str(group[c["target"]][0])
        events = parse_session(group)
        if events:
            yield str(participant), target, events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aalto_adapter.py -v`
Expected: 7 passed. If the first test fails, correct `AALTO_COLUMNS` to the printed actual column names and re-run.

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/adapters tests/test_aalto_adapter.py tests/fixtures/aalto_sample.csv
git commit -m "feat: Aalto corpus adapter"
```

---

### Task 8: KLiCKe adapter and the round-trip milestone

**Files:**
- Create: `src/typeshi/adapters/klicke.py`, `tests/fixtures/klicke_sample.csv`
- Test: `tests/test_klicke_adapter.py`

**Interfaces:**
- Consumes: `Event`, `TextBuffer`, `replay`, `serialize`, `deserialize`
- Produces: `KLICKE_COLUMNS: dict[str, str]` (canonical → real names for `writer`, `event_type`, `timestamp_ms`, `char`, `cursor_pos`, `text_change`); `parse_session(rows) -> list[Event]`; `iter_sessions(path) -> Iterator[tuple[str, str, list[Event]]]` yielding `(writer_id, final_text, events)`.

This is the milestone task: KLiCKe logs full non-linear editing, so a correct adapter must reproduce the final essay text exactly by replay. That is the spec's "round-trip replay" gate.

- [ ] **Step 1: Write the fixture and failing test**

Create `tests/fixtures/klicke_sample.csv` from one complete short real session (header verbatim), including at least one cursor jump and one deletion.

```python
# tests/test_klicke_adapter.py
from pathlib import Path

import polars as pl
import pytest

from typeshi.adapters.klicke import KLICKE_COLUMNS, iter_sessions
from typeshi.buffer import replay
from typeshi.events import EventType
from typeshi.serialize import deserialize, serialize

FIXTURE = Path(__file__).parent / "fixtures" / "klicke_sample.csv"


def test_fixture_has_the_columns_the_adapter_expects():
    df = pl.read_csv(FIXTURE, n_rows=1)
    missing = [c for c in KLICKE_COLUMNS.values() if c not in df.columns]
    assert not missing, f"missing {missing}; actual columns are {df.columns}"


def test_session_contains_nonlinear_editing():
    _, _, events = next(iter_sessions(FIXTURE))
    types = {e.type for e in events}
    assert EventType.CURSOR in types or EventType.SELDEL in types


def test_replay_reproduces_the_final_text_exactly():
    """THE MILESTONE: a real logged session round-trips byte for byte."""
    _, final_text, events = next(iter_sessions(FIXTURE))
    assert replay(events) == final_text


def test_serialization_round_trip_preserves_the_final_text():
    """Events -> tokens -> events must still produce the same essay."""
    _, final_text, events = next(iter_sessions(FIXTURE))
    assert replay(deserialize(serialize(events))) == final_text


def test_timestamps_are_monotonic_and_zero_based():
    _, _, events = next(iter_sessions(FIXTURE))
    assert events[0].press_time == 0
    times = [e.press_time for e in events]
    assert times == sorted(times)


def test_long_thinking_pauses_survive_binning():
    """Composition has multi-second pauses; binning must not flatten them."""
    _, _, events = next(iter_sessions(FIXTURE))
    gaps = [b.press_time - a.press_time for a, b in zip(events, events[1:])]
    restored = deserialize(serialize(events))
    rgaps = [b.press_time - a.press_time for a, b in zip(restored, restored[1:])]
    assert max(rgaps) > 0.85 * max(gaps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_klicke_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.adapters.klicke'`

- [ ] **Step 3: Write the adapter**

Fill `KLICKE_COLUMNS` from `docs/data-schemas.md`. KLiCKe logs an event type per row plus cursor position; emit a `CURSOR` event only when the logged position differs from where the buffer cursor already sits, so we do not spam redundant moves.

```python
# src/typeshi/adapters/klicke.py
"""KLiCKe corpus -> canonical events.

Composition logs are non-linear: cursor jumps, range deletions, insertions
mid-text. Correctness bar is exact replay of the final essay.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl

from typeshi.buffer import TextBuffer
from typeshi.events import Event

# Canonical name -> real CSV column. Verified against docs/data-schemas.md.
KLICKE_COLUMNS = {
    "writer": "id",
    "event_type": "type",
    "timestamp_ms": "down_time",
    "release_ms": "up_time",
    "char": "text_change",
    "cursor_pos": "cursor_position",
}

_INSERT = {"input", "insert", "keydown"}
_DELETE = {"remove/cut", "delete", "backspace"}


def parse_session(rows: pl.DataFrame) -> tuple[str, list[Event]]:
    """Returns (final_text, events). Emits cursor moves only when needed."""
    c = KLICKE_COLUMNS
    rows = rows.sort(c["timestamp_ms"])
    if rows.is_empty():
        return "", []
    t0 = int(rows[c["timestamp_ms"]][0])

    buf = TextBuffer()
    events: list[Event] = []

    for row in rows.iter_rows(named=True):
        press = int(row[c["timestamp_ms"]]) - t0
        release_raw = row.get(c["release_ms"])
        release = max(int(release_raw) - t0, press) if release_raw is not None else press
        kind = str(row[c["event_type"]]).strip().lower()
        change = row[c["char"]]
        pos = row[c["cursor_pos"]]

        # Reposition first if the log says the cursor is elsewhere.
        if pos is not None:
            target_pos = int(pos)
            if kind in _INSERT:
                target_pos = max(target_pos - len(str(change or "")), 0)
            target_pos = min(target_pos, len(buf.text))
            if target_pos != buf.cursor:
                move = Event.cursor(target_pos, press)
                events.append(move)
                buf.apply(move)

        if kind in _INSERT and change is not None:
            for ch in str(change):
                e = Event.key(ch, press, release)
                events.append(e)
                buf.apply(e)
        elif kind in _DELETE:
            n = len(str(change)) if change is not None else 1
            if n == 1:
                e = Event.backspace(press, release)
                events.append(e)
                buf.apply(e)
            elif n > 1:
                start = max(buf.cursor - n, 0)
                if start < buf.cursor:
                    e = Event.seldel(start, buf.cursor, press)
                    events.append(e)
                    buf.apply(e)
        # Other row kinds (mouse moves, focus changes) carry no text effect.

    return buf.text, events


def iter_sessions(path: Path) -> Iterator[tuple[str, str, list[Event]]]:
    c = KLICKE_COLUMNS
    df = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)
    for (writer,), group in df.group_by([c["writer"]], maintain_order=True):
        final_text, events = parse_session(group)
        if events:
            yield str(writer), final_text, events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_klicke_adapter.py -v`
Expected: 6 passed.

If `test_replay_reproduces_the_final_text_exactly` fails, the log's event-kind vocabulary differs from `_INSERT`/`_DELETE` above, or cursor positions are recorded post-edit rather than pre-edit. Print the diff between `replay(events)` and `final_text` and adjust the offset logic — do **not** relax the assertion. Exact replay is the gate.

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/adapters/klicke.py tests/test_klicke_adapter.py tests/fixtures/klicke_sample.csv
git commit -m "feat: KLiCKe adapter with exact round-trip replay"
```

---

### Task 9: Windowing and JSONL training export

**Files:**
- Create: `src/typeshi/dataset.py`, `scripts/build_dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: adapters, `serialize`, `compute_labels`, `SessionLabels`
- Produces: `build_prompt(target_text, labels, mode, written_so_far="", cursor=None) -> str`; `build_examples(target_text, events, labels, mode, max_events=512) -> list[dict]` where each dict is `{"prompt": str, "completion": str}`; `split_by_writer(writer_ids, test_frac=0.1, seed=0) -> tuple[set[str], set[str]]`; CLI `scripts/build_dataset.py` writing `data/processed/{train,test}.jsonl`.

Long essays exceed the context window, so sessions are cut into windows of at most `max_events` events. Each window's prompt carries the target text, knob header, and a state header (buffer prefix summary and cursor position) so the model can resume mid-session.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset.py
import pytest
from typeshi.dataset import build_examples, split_by_writer
from typeshi.events import Event
from typeshi.labels import SessionLabels

LABELS = SessionLabels(60.0, 0.02, 0.0, 0.05)


def _type(text, gap=100):
    return [Event.key(c, i * gap, i * gap + 50) for i, c in enumerate(text)]


def test_short_session_becomes_one_example():
    ex = build_examples("hello", _type("hello"), LABELS, mode="transcription")
    assert len(ex) == 1
    assert set(ex[0]) == {"prompt", "completion"}


def test_prompt_contains_target_text_and_knobs():
    ex = build_examples("hello", _type("hello"), LABELS, mode="transcription")
    assert "hello" in ex[0]["prompt"]
    assert "WPM=60" in ex[0]["prompt"]
    assert "MODE=transcription" in ex[0]["prompt"]


def test_completion_is_the_event_token_stream():
    ex = build_examples("hi", _type("hi"), LABELS, mode="transcription")
    assert "<KEY:h>" in ex[0]["completion"]
    assert "<DT:" in ex[0]["completion"]


def test_long_session_is_split_into_windows():
    events = _type("a" * 1200)
    ex = build_examples("a" * 1200, events, LABELS, mode="composition", max_events=512)
    assert len(ex) == 3


def test_continuation_windows_carry_resume_state():
    events = _type("a" * 1200)
    ex = build_examples("a" * 1200, events, LABELS, mode="composition", max_events=512)
    assert "CURSOR=" in ex[1]["prompt"]
    assert "CURSOR=" not in ex[0]["prompt"]


def test_windows_cover_every_event_exactly_once():
    events = _type("a" * 1000)
    ex = build_examples("a" * 1000, events, LABELS, mode="composition", max_events=512)
    total_keys = sum(e["completion"].count("<KEY:") for e in ex)
    assert total_keys == 1000


def test_split_is_by_writer_and_deterministic():
    writers = [f"w{i}" for i in range(100)]
    train_a, test_a = split_by_writer(writers, test_frac=0.1, seed=0)
    train_b, test_b = split_by_writer(writers, test_frac=0.1, seed=0)
    assert (train_a, test_a) == (train_b, test_b)
    assert not (train_a & test_a)
    assert len(test_a) == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.dataset'`

- [ ] **Step 3: Write the implementation**

```python
# src/typeshi/dataset.py
"""Turns parsed sessions into prompt/completion training examples."""

from __future__ import annotations

import random
from typing import Iterable

from typeshi.buffer import TextBuffer
from typeshi.events import Event
from typeshi.labels import SessionLabels
from typeshi.serialize import serialize

_PROMPT = (
    "Simulate the writing process for the target text.\n"
    "{header}\n"
    "TARGET: {target}\n"
    "{state}"
    "PROCESS:"
)


def build_prompt(
    target_text: str,
    labels: SessionLabels,
    mode: str,
    written_so_far: str = "",
    cursor: int | None = None,
) -> str:
    """The prompt format shared by training export and inference."""
    state = ""
    if cursor is not None:
        # Resume state: how far along the writer is, and where the caret sits.
        state = f"WRITTEN_SO_FAR: {written_so_far}\nCURSOR={cursor}\n"
    return _PROMPT.format(header=labels.to_header(mode), target=target_text, state=state)


def build_examples(
    target_text: str,
    events: list[Event],
    labels: SessionLabels,
    mode: str,
    max_events: int = 512,
) -> list[dict]:
    examples: list[dict] = []
    buf = TextBuffer()

    for start in range(0, len(events), max_events):
        window = events[start : start + max_events]
        prompt = (
            build_prompt(target_text, labels, mode)
            if start == 0
            else build_prompt(target_text, labels, mode, buf.text, buf.cursor)
        )
        examples.append({"prompt": prompt, "completion": serialize(window)})
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
```

```python
# scripts/build_dataset.py
"""Parses both corpora into data/processed/{train,test}.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typeshi.adapters import aalto, klicke
from typeshi.dataset import build_examples, split_by_writer
from typeshi.labels import compute_labels


def collect(path: Path, module, mode: str) -> list[tuple[str, dict]]:
    out = []
    for data_file in sorted(path.rglob("*.csv")):
        for writer, target, events in module.iter_sessions(data_file):
            labels = compute_labels(events, target)
            for ex in build_examples(target, events, labels, mode=mode):
                out.append((writer, ex))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aalto", type=Path, default=Path("data/raw/aalto"))
    ap.add_argument("--klicke", type=Path, default=Path("data/raw/klicke"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    if args.aalto.exists():
        rows += collect(args.aalto, aalto, "transcription")
    if args.klicke.exists():
        rows += collect(args.klicke, klicke, "composition")

    train_ids, test_ids = split_by_writer([w for w, _ in rows], seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    for name, ids in (("train", train_ids), ("test", test_ids)):
        path = args.out / f"{name}.jsonl"
        with path.open("w") as fh:
            n = 0
            for writer, ex in rows:
                if writer in ids:
                    fh.write(json.dumps(ex) + "\n")
                    n += 1
        print(f"wrote {n} examples to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dataset.py -v`
Expected: 7 passed

Then build the real dataset: `uv run python scripts/build_dataset.py`

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/dataset.py scripts/build_dataset.py tests/test_dataset.py
git commit -m "feat: session windowing and JSONL dataset export"
```

---

### Task 10: Phase-1 motor LoRA fine-tune

**Files:**
- Create: `src/typeshi/train_motor.py`
- Test: `tests/test_train_motor.py`

**Interfaces:**
- Consumes: `special_tokens()`, `config.BASE_MODEL`, `data/processed/train.jsonl`
- Produces: `prepare_tokenizer(base_model) -> tokenizer` (with special tokens added), `build_peft_config() -> LoraConfig`, CLI training entry point writing an adapter to `checkpoints/motor/`.

Tests cover the tokenizer wiring only — the training run itself is verified by the eval harness in Tasks 11–12, not by unit tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_motor.py
import pytest

transformers = pytest.importorskip("transformers")

from typeshi.serialize import special_tokens
from typeshi.train_motor import build_peft_config, prepare_tokenizer

TINY = "hf-internal-testing/tiny-random-gpt2"


def test_special_tokens_are_added_to_the_vocabulary():
    tok = prepare_tokenizer(TINY)
    for t in ["<BKSP>", "<DT:0>", "<KEY:SPC>"]:
        assert len(tok.tokenize(t)) == 1, f"{t} was split into multiple tokens"


def test_event_tokens_survive_encode_decode():
    tok = prepare_tokenizer(TINY)
    s = "<DT:5><KEY:a><HOLD:3><BKSP>"
    assert tok.decode(tok.encode(s), skip_special_tokens=False).replace(" ", "") == s


def test_vocabulary_grew_by_the_expected_amount():
    base = transformers.AutoTokenizer.from_pretrained(TINY)
    tok = prepare_tokenizer(TINY)
    assert len(tok) >= len(base) + len(special_tokens()) - 3  # 3 are prefixes, not whole tokens


def test_peft_config_targets_attention_projections():
    cfg = build_peft_config()
    assert cfg.r >= 16
    assert "q_proj" in cfg.target_modules and "v_proj" in cfg.target_modules
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_motor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.train_motor'`

- [ ] **Step 3: Write the implementation**

```python
# src/typeshi/train_motor.py
"""Phase 1: LoRA fine-tune on transcription data to learn motor timing."""

from __future__ import annotations

import argparse
from pathlib import Path

from typeshi import config
from typeshi.serialize import special_tokens


def prepare_tokenizer(base_model: str = config.BASE_MODEL):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    # Prefix-only entries ("<CUR:", "<SELDEL:") carry variable integers, so only
    # whole tokens are registered; the integers tokenize as ordinary digits.
    whole = [t for t in special_tokens() if t.endswith(">")]
    tok.add_tokens(whole, special_tokens=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_peft_config():
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def main() -> None:
    import torch
    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/motor"))
    ap.add_argument("--base", default=config.BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mode", default="transcription",
                    help="filter examples by MODE= in the prompt")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = ap.parse_args()

    tok = prepare_tokenizer(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.resize_token_embeddings(len(tok))
    model = get_peft_model(model, build_peft_config())
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(args.data), split="train")
    ds = ds.filter(lambda r: f"MODE={args.mode}" in r["prompt"])
    ds = ds.map(lambda r: {"text": r["prompt"] + r["completion"] + tok.eos_token})

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        processing_class=tok,
        args=SFTConfig(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            bf16=True,
            logging_steps=25,
            save_strategy="epoch",
            seed=args.seed,
            max_length=2048,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_motor.py -v`
Expected: 4 passed

Then launch the real run on the rented GPU:
`uv run python -m typeshi.train_motor --mode transcription --out checkpoints/motor`

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/train_motor.py tests/test_train_motor.py
git commit -m "feat: phase-1 motor LoRA training"
```

---

### Task 11: Generation harness and distributional evaluation

**Files:**
- Create: `src/typeshi/generate.py`, `src/typeshi/eval/__init__.py`, `src/typeshi/eval/distributional.py`
- Test: `tests/test_distributional.py`

**Interfaces:**
- Consumes: `deserialize`, `replay`, `Event`, `EventType`
- Produces: `generate(model, tok, target_text, labels, mode, temperature=1.0, seed=0) -> list[Event]`; `timing_features(events) -> dict[str, np.ndarray]` with keys `iki`, `hold`, `pause`, `burst`; `kl_divergence(p_samples, q_samples, bins=64) -> float`; `frechet_distance(p_samples, q_samples) -> float`; `compare(real_events, fake_events) -> dict[str, dict[str, float]]`.

`iki` is press-to-press gaps, `hold` is per-key durations, `pause` is gaps above 1000 ms, `burst` is run lengths of consecutive keys between pauses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_distributional.py
import numpy as np
import pytest

from typeshi.events import Event
from typeshi.eval.distributional import (
    compare, frechet_distance, kl_divergence, timing_features,
)


def _session(gaps, hold=50):
    events, t = [], 0
    for g in gaps:
        events.append(Event.key("a", t, t + hold))
        t += g
    return events


def test_timing_features_extracts_iki_and_hold():
    f = timing_features(_session([100, 120, 140]))
    assert list(f["iki"]) == [100, 120]
    assert list(f["hold"]) == [50, 50, 50]


def test_pauses_are_gaps_over_one_second():
    f = timing_features(_session([100, 5000, 100, 3000]))
    assert sorted(f["pause"]) == [3000, 5000]


def test_bursts_are_key_runs_between_pauses():
    f = timing_features(_session([100, 100, 5000, 100, 100, 100]))
    assert sorted(f["burst"]) == [3, 4]


def test_kl_of_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.lognormal(5, 0.4, 4000)
    y = rng.lognormal(5, 0.4, 4000)
    assert kl_divergence(x, y) < 0.05


def test_kl_grows_when_distributions_differ():
    rng = np.random.default_rng(0)
    x = rng.lognormal(5, 0.4, 4000)
    y = rng.lognormal(6, 0.4, 4000)
    assert kl_divergence(x, y) > kl_divergence(x, x + 1)


def test_frechet_is_zero_for_identical_samples():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert frechet_distance(x, x) == pytest.approx(0.0, abs=1e-9)


def test_compare_reports_every_feature():
    real, fake = _session([100] * 50), _session([110] * 50)
    result = compare(real, fake)
    assert set(result) == {"iki", "hold", "pause", "burst"}
    assert "kl" in result["iki"] and "frechet" in result["iki"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_distributional.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.eval'`

- [ ] **Step 3: Write the implementation**

```python
# src/typeshi/eval/distributional.py
"""Distributional fidelity metrics between real and generated timing."""

from __future__ import annotations

import numpy as np

from typeshi.events import Event, EventType

PAUSE_THRESHOLD_MS = 1000


def timing_features(events: list[Event]) -> dict[str, np.ndarray]:
    press = np.array([e.press_time for e in events], dtype=float)
    iki = np.diff(press) if len(press) > 1 else np.array([])
    hold = np.array(
        [e.release_time - e.press_time for e in events if e.release_time is not None],
        dtype=float,
    )
    pause = iki[iki > PAUSE_THRESHOLD_MS] if iki.size else np.array([])

    bursts, run = [], 1
    for gap in iki:
        if gap > PAUSE_THRESHOLD_MS:
            bursts.append(run)
            run = 1
        else:
            run += 1
    bursts.append(run)

    return {"iki": iki, "hold": hold, "pause": pause,
            "burst": np.array(bursts, dtype=float)}


def kl_divergence(p_samples, q_samples, bins: int = 64) -> float:
    """Symmetrized KL over a shared log-spaced histogram."""
    p_samples = np.asarray(p_samples, dtype=float)
    q_samples = np.asarray(q_samples, dtype=float)
    if p_samples.size == 0 or q_samples.size == 0:
        return float("nan")

    lo = max(min(p_samples.min(), q_samples.min()), 1e-3)
    hi = max(p_samples.max(), q_samples.max())
    edges = np.geomspace(lo, hi + 1, bins + 1)

    p, _ = np.histogram(p_samples, bins=edges)
    q, _ = np.histogram(q_samples, bins=edges)
    p = (p + 1e-9) / (p.sum() + 1e-9 * bins)
    q = (q + 1e-9) / (q.sum() + 1e-9 * bins)
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def frechet_distance(p_samples, q_samples) -> float:
    """1-D Frechet distance: (mu1-mu2)^2 + (s1-s2)^2, on log-transformed times."""
    p = np.log1p(np.asarray(p_samples, dtype=float))
    q = np.log1p(np.asarray(q_samples, dtype=float))
    if p.size == 0 or q.size == 0:
        return float("nan")
    return float((p.mean() - q.mean()) ** 2 + (p.std() - q.std()) ** 2)


def compare(real_events: list[Event], fake_events: list[Event]) -> dict:
    real, fake = timing_features(real_events), timing_features(fake_events)
    return {
        key: {
            "kl": kl_divergence(real[key], fake[key]),
            "frechet": frechet_distance(real[key], fake[key]),
        }
        for key in real
    }
```

```python
# src/typeshi/generate.py
"""Sampling harness: prompt the fine-tuned model, parse events back out."""

from __future__ import annotations

from typeshi.dataset import build_prompt
from typeshi.events import Event
from typeshi.labels import SessionLabels
from typeshi.serialize import deserialize


def generate(
    model,
    tok,
    target_text: str,
    labels: SessionLabels,
    mode: str = "transcription",
    temperature: float = 1.0,
    max_new_tokens: int = 4096,
    seed: int = 0,
) -> list[Event]:
    import torch

    torch.manual_seed(seed)
    prompt = build_prompt(target_text, labels, mode)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
    )
    completion = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=False)
    completion = completion.replace(tok.eos_token or "", "").replace(" ", "")
    return deserialize(completion)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_distributional.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/generate.py src/typeshi/eval tests/test_distributional.py
git commit -m "feat: generation harness and distributional eval metrics"
```

---

### Task 12: Real-vs-fake discriminator (the Turing test)

**Files:**
- Create: `src/typeshi/eval/discriminator.py`, `scripts/run_eval.py`
- Test: `tests/test_discriminator.py`

**Interfaces:**
- Consumes: `timing_features`, `Event`
- Produces: `featurize(events) -> np.ndarray` (fixed-length summary vector); `train_discriminator(real_sessions, fake_sessions, seed=0) -> (clf, accuracy)`; `heuristic_baseline(target_text, wpm, seed=0) -> list[Event]` (a deliberately naive Gaussian-jitter simulator used to validate the discriminator itself).

The gate: the discriminator should score **≥ 0.9 accuracy against the heuristic baseline** (proving it has teeth) and **≤ 0.55 against our model** (proving our output is indistinguishable). A gradient-boosted classifier over summary statistics is the v1 discriminator; a sequence model can replace it later without changing this interface.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_discriminator.py
import numpy as np
import pytest

from typeshi.events import Event
from typeshi.eval.discriminator import (
    featurize, heuristic_baseline, train_discriminator,
)


def _lognormal_session(rng, n=120, mu=4.8, sigma=0.45):
    events, t = [], 0
    for _ in range(n):
        events.append(Event.key("a", t, t + int(rng.lognormal(4.0, 0.3))))
        t += int(rng.lognormal(mu, sigma))
    return events


def test_featurize_returns_fixed_length_vector():
    rng = np.random.default_rng(0)
    a = featurize(_lognormal_session(rng))
    b = featurize(_lognormal_session(rng, n=300))
    assert a.shape == b.shape
    assert np.isfinite(a).all()


def test_heuristic_baseline_produces_the_target_text():
    from typeshi.buffer import replay
    events = heuristic_baseline("hello world", wpm=60, seed=0)
    assert replay(events) == "hello world"


def test_discriminator_easily_catches_the_heuristic_baseline():
    """Validates the discriminator has teeth before we trust its verdict."""
    rng = np.random.default_rng(0)
    real = [_lognormal_session(rng) for _ in range(60)]
    fake = [heuristic_baseline("the quick brown fox jumps", wpm=60, seed=i)
            for i in range(60)]
    _, acc = train_discriminator(real, fake, seed=0)
    assert acc > 0.9


def test_discriminator_cannot_separate_identical_distributions():
    """Sanity check: same generator on both sides -> chance accuracy."""
    rng = np.random.default_rng(0)
    a = [_lognormal_session(rng) for _ in range(60)]
    b = [_lognormal_session(rng) for _ in range(60)]
    _, acc = train_discriminator(a, b, seed=0)
    assert acc < 0.65
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_discriminator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.eval.discriminator'`

- [ ] **Step 3: Write the implementation**

```python
# src/typeshi/eval/discriminator.py
"""Learned real-vs-generated classifier. Our pass condition is that this
model CANNOT tell the difference, so it must first be shown to have teeth."""

from __future__ import annotations

import numpy as np

from typeshi.events import Event
from typeshi.eval.distributional import timing_features

_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def featurize(events: list[Event]) -> np.ndarray:
    f = timing_features(events)
    parts: list[float] = []
    for key in ("iki", "hold", "pause", "burst"):
        x = f[key]
        if x.size == 0:
            parts += [0.0] * (len(_QUANTILES) + 3)
            continue
        lx = np.log1p(x)
        parts += list(np.quantile(lx, _QUANTILES))
        parts += [float(lx.mean()), float(lx.std()), float(len(x))]
    # Autocorrelation of successive gaps: humans drift, naive samplers do not.
    iki = f["iki"]
    if iki.size > 2:
        li = np.log1p(iki)
        parts.append(float(np.corrcoef(li[:-1], li[1:])[0, 1]))
    else:
        parts.append(0.0)
    return np.nan_to_num(np.array(parts, dtype=float))


def heuristic_baseline(target_text: str, wpm: float, seed: int = 0) -> list[Event]:
    """Deliberately naive simulator: Gaussian jitter around a fixed mean gap.
    Stands in for off-the-shelf typing simulators as a discriminator control."""
    rng = np.random.default_rng(seed)
    mean_gap = 60_000 / (wpm * 5)
    events, t = [], 0
    for ch in target_text:
        gap = max(int(rng.normal(mean_gap, mean_gap * 0.15)), 1)
        hold = max(int(rng.normal(80, 12)), 1)
        events.append(Event.key(ch, t, t + hold))
        t += gap
    return events


def train_discriminator(
    real_sessions: list[list[Event]],
    fake_sessions: list[list[Event]],
    seed: int = 0,
) -> tuple[object, float]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    X = np.vstack(
        [featurize(s) for s in real_sessions] + [featurize(s) for s in fake_sessions]
    )
    y = np.array([1] * len(real_sessions) + [0] * len(fake_sessions))
    clf = GradientBoostingClassifier(random_state=seed)
    acc = float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean())
    clf.fit(X, y)
    return clf, acc
```

Add `scikit-learn>=1.4` to the `dev` and base dependencies in `pyproject.toml`.

```python
# scripts/run_eval.py
"""Scores a trained checkpoint: distributional metrics + discriminator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typeshi.adapters import aalto
from typeshi.eval.discriminator import heuristic_baseline, train_discriminator
from typeshi.eval.distributional import compare
from typeshi.generate import generate
from typeshi.labels import compute_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/motor"))
    ap.add_argument("--held-out", type=Path, default=Path("data/raw/aalto"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    args = ap.parse_args()

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint, device_map="auto")

    real, fake, baseline = [], [], []
    for i, (_writer, target, events) in enumerate(
        aalto.iter_sessions(next(args.held_out.rglob("*.csv")))
    ):
        if i >= args.n:
            break
        labels = compute_labels(events, target)
        real.append(events)
        fake.append(generate(model, tok, target, labels, mode="transcription", seed=i))
        baseline.append(heuristic_baseline(target, wpm=labels.wpm, seed=i))

    flat = lambda ss: [e for s in ss for e in s]  # noqa: E731
    _, acc_model = train_discriminator(real, fake)
    _, acc_baseline = train_discriminator(real, baseline)

    report = {
        "distributional": compare(flat(real), flat(fake)),
        "discriminator_accuracy_vs_model": acc_model,
        "discriminator_accuracy_vs_heuristic_baseline": acc_baseline,
        "pass_model": acc_model <= 0.55,
        "pass_discriminator_has_teeth": acc_baseline >= 0.9,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_discriminator.py -v`
Expected: 4 passed

Then score the trained checkpoint: `uv run python scripts/run_eval.py --checkpoint checkpoints/motor`

**Tier-1 milestone is met when** `pass_discriminator_has_teeth` is true and `discriminator_accuracy_vs_model` is at or below 0.55. If the model fails, the levers in order are: more training data, lower sampling temperature, longer training, then a sequence-model discriminator to find what the summary-statistic one is missing.

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/eval/discriminator.py scripts/run_eval.py tests/test_discriminator.py pyproject.toml
git commit -m "feat: real-vs-fake discriminator and eval report"
```

---

## Plan Self-Review Notes

**Spec coverage for build-order steps 1–2:** event serialization (§3) → Tasks 1, 3, 4; data sources and preparation (§4) → Tasks 6, 7, 8, 9; writer-level splits (§4.2) → Task 9; condition labels/knobs (§7) → Task 5; Phase-1 motor training (§5.1) → Task 10; distributional + discriminator eval (§8.1, §8.2) → Tasks 11, 12; round-trip milestone (§11.1) → Task 8; Tier-1 milestone (§11.2) → Task 12. License verification (§10) → Task 6 Step 2.

**Deferred to later plans (by design):** Phase-2 composition fine-tune, constrained decoding and the convergence guarantee, synthetic backward draft chains, knob-fidelity and composition-signature evals, and Modal/Baseten serving.

**Known follow-ups this plan intentionally leaves open:** the discriminator is summary-statistic based (a sequence model is the upgrade path if it proves too weak); `<CUR:` and `<SELDEL:` integers tokenize as digits rather than single vocabulary entries, which is fine for Phase 1 since transcription emits neither.
