"""KLiCKe corpus -> canonical events.

Composition logs are non-linear: cursor jumps, range deletions, insertions
mid-text. Correctness bar is exact replay of the final essay, checked against
the essay text shipped alongside the log rather than against our own output.

Schema notes that drive this file (see docs/data-schemas.md):
  - the writer ID is the *filename*; there is no participant column
  - `CursorPosition` is the caret offset *after* the row's edit is applied
  - `TextChange` carries backslash escapes (`\\n` is two characters)
  - some logs contain invalid UTF-8, so reads are lossy
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import polars as pl

from typeshi.buffer import ReplayError, TextBuffer
from typeshi.events import Event, rebase

# Canonical name -> real CSV column. Verified against docs/data-schemas.md.
KLICKE_COLUMNS = {
    "event_type": "Activity",
    "timestamp_ms": "DownTime",
    "release_ms": "UpTime",
    "char": "TextChange",
    "cursor_pos": "CursorPosition",
    "key": "DownEvent",
}

_INSERT = {"input"}
_DELETE = {"remove/cut"}
_REPLACE = "replace"
_MOVE_RE = re.compile(r"Move From \[(\d+),\s*(\d+)\] To \[(\d+),\s*(\d+)\]")
_UNESCAPE = ((r"\n", "\n"), (r"\"", '"'), (r"\\", "\\"))


class UnreplayableSession(Exception):
    """The log's positions are self-inconsistent, so it cannot be replayed."""


class UnsupportedSession(UnreplayableSession):
    """The session contains operations the motor model must not learn from.

    Paste and drag-drop expand into runs of KEY events sharing one timestamp
    -- measured 142 paste rows becoming 15,031 zero-IKI "keystrokes" in 500
    logs. Training on those teaches instant multi-character typing, which is
    exactly the failure mode the model exists to avoid. 13.8% of WritingTask
    sessions contain a paste; they are dropped, not patched.
    """


def _unescape(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "NoChange":
        return ""
    for escaped, literal in _UNESCAPE:
        text = text.replace(escaped, literal)
    return text


def _emit(events: list[Event], buf: TextBuffer, event: Event) -> None:
    events.append(event)
    buf.apply(event)


def _seek(events: list[Event], buf: TextBuffer, pos: int, press: int) -> None:
    """Move the caret to `pos`, emitting a CURSOR event only if it actually moved."""
    if pos == buf.cursor:
        return
    if not 0 <= pos <= len(buf.text):
        raise UnreplayableSession(f"cursor {pos} outside buffer of {len(buf.text)}")
    _emit(events, buf, Event.cursor(pos, press))


def _read(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path, infer_schema_length=5000, ignore_errors=True, encoding="utf8-lossy"
    )


def parse_session(rows: pl.DataFrame) -> tuple[str, list[Event]]:
    """Returns (replayed_text, events). Emits cursor moves only when needed.

    Raises UnreplayableSession if the log's positions are inconsistent, which
    happens when key rollover makes the logger record a stale CursorPosition.
    """
    c = KLICKE_COLUMNS
    rows = rows.sort(c["timestamp_ms"])
    if rows.is_empty():
        return "", []
    t0 = int(rows[c["timestamp_ms"]][0])

    buf = TextBuffer()
    events: list[Event] = []
    # Navigation rows (arrows, clicks) carry no text change but DO carry the
    # moment the caret actually moved. Without this, a cursor jump at second 1
    # followed by typing at second 10 was stamped at second 10, relocating the
    # nine-second pause to the wrong side of the move -- 85,970 such rows in
    # the first 500 logs. Remember the last navigated position and use its
    # own timestamp when the next edit confirms it.
    nav: tuple[int, int] | None = None

    def seek_time(target_pos: int, edit_press: int) -> int:
        if nav is not None and nav[0] == target_pos:
            return nav[1]
        return edit_press

    for row in rows.iter_rows(named=True):
        activity, pos = row[c["event_type"]], row[c["cursor_pos"]]
        if activity is None or pos is None:
            continue

        press = int(row[c["timestamp_ms"]]) - t0
        release_raw = row.get(c["release_ms"])
        release = max(int(release_raw) - t0, press) if release_raw is not None else press
        kind = str(activity).strip().lower()
        change = _unescape(row[c["char"]])
        pos = int(pos)

        if kind == "nonproduction":
            if pos != buf.cursor and 0 <= pos <= len(buf.text):
                nav = (pos, press)
            continue

        if kind == "paste" or kind.startswith("move"):
            raise UnsupportedSession(f"session contains {kind!r}")

        if kind in _INSERT:
            # CursorPosition is post-edit, so the text went in len(change) earlier.
            target = pos - len(change)
            _seek(events, buf, target, seek_time(target, press))
            for ch in change:
                _emit(events, buf, Event.key(ch, press, release))

        elif kind in _DELETE:
            # Post-edit caret sits at the start of the removed span, for both
            # Backspace and Delete.
            n = len(change)
            if n == 0:
                continue
            if pos + n > len(buf.text):
                raise UnreplayableSession(f"delete {pos}+{n} past {len(buf.text)}")
            if n == 1:
                _seek(events, buf, pos + 1, seek_time(pos + 1, press))
                _emit(events, buf, Event.backspace(press, release))
            else:
                _seek(events, buf, pos, seek_time(pos, press))
                _emit(events, buf, Event.seldel(pos, pos + n, press))

        elif kind == _REPLACE:
            # "<old> => <new>": typed over a selection. Split on the last
            # separator because the replaced text may itself contain " => ".
            if " => " not in change:
                raise UnreplayableSession("malformed Replace payload")
            old, new = change.rsplit(" => ", 1)
            start = pos - len(new)
            if start < 0 or start + len(old) > len(buf.text):
                raise UnreplayableSession("Replace span outside buffer")
            if old:
                _seek(events, buf, start, seek_time(start, press))
                _emit(events, buf, Event.seldel(start, start + len(old), press))
            _seek(events, buf, start, press)
            for ch in new:
                _emit(events, buf, Event.key(ch, press, release))


        nav = None

    return buf.text, rebase(events)


def gold_text_path(csv_path: Path) -> Path | None:
    """Locates the corpus's final-essay file for a keystroke log.

    The corpus stores essays under `WritingTask/texts/<writer>.txt`, a sibling
    of the `csv/` directory; test fixtures keep the .txt next to the .csv.
    """
    candidates = [
        csv_path.with_suffix(".txt"),
        csv_path.parent.parent / "texts" / f"{csv_path.stem}.txt",
        csv_path.parent.parent.parent / "texts" / f"{csv_path.stem}.txt",
    ]
    return next((p for p in candidates if p.exists()), None)


def iter_sessions(path: Path) -> Iterator[tuple[str, str, list[Event]]]:
    """Yields (writer_id, final_text, events) for each log under `path`.

    `final_text` is the corpus's own essay file. Sessions whose replay does not
    reproduce it byte for byte are dropped — roughly 4.5% of the corpus, where
    rollover corrupted the logged cursor positions. Dropping rather than
    patching keeps exact replay as the correctness gate.
    """
    path = Path(path)
    files = sorted(path.rglob("*.csv")) if path.is_dir() else [path]

    for csv_path in files:
        gold_path = gold_text_path(csv_path)
        if gold_path is None:
            continue
        gold = gold_path.read_text(encoding="utf-8", errors="replace")

        try:
            produced, events = parse_session(_read(csv_path))
        except (UnreplayableSession, ReplayError):
            continue
        # Trailing newlines are normalised on both sides: writers really do end
        # with blank lines, and the text exporter adds one more on top. Every
        # other character must match exactly.
        if events and produced.rstrip("\n") == gold.rstrip("\n"):
            yield csv_path.stem, produced, events
