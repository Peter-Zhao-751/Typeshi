"""Aalto 136M Keystrokes -> canonical events.

Transcription only: linear typing plus backspaces. No cursor movement.

Schema notes that drive this file (see docs/data-schemas.md):
  - files are TAB-separated `<participant>_keystrokes.txt`, one participant each
  - `PRESS_TIME` / `RELEASE_TIME` are Unix epoch ms, so sessions are rebased
  - `LETTER` is the literal character, or a name (`BKSP`, `SHIFT`, `CAPS_LOCK`)
  - `LETTER` is null on ~8% of rows where the logger failed to record the key
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl

from typeshi.events import Event, rebase

# Canonical name -> real column. Verified against the corpus readme and header.
AALTO_COLUMNS = {
    "participant": "PARTICIPANT_ID",
    "session": "TEST_SECTION_ID",
    "target": "SENTENCE",
    "user_input": "USER_INPUT",
    "press_ms": "PRESS_TIME",
    "release_ms": "RELEASE_TIME",
    "char": "LETTER",
    "keycode": "KEYCODE",
}

_BACKSPACE_MARKERS = {"BKSP", "BACKSPACE", "\x08"}

# Physical keyboards only; 'small' and 'on-screen' are mobile and out of scope.
PHYSICAL_KEYBOARDS = {"full", "laptop"}


def read_log(path: Path) -> pl.DataFrame:
    """Reads one keystroke log.

    `quote_char=None` matters: sentences contain apostrophes and quotation
    marks that would otherwise be treated as field delimiters and shear rows
    apart.
    """
    return pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10_000,
        ignore_errors=True,
        encoding="utf8-lossy",
        quote_char=None,
        truncate_ragged_lines=True,
    )


def parse_session(rows: pl.DataFrame) -> list[Event]:
    """One test section -> events, with times rebased to the first keystroke."""
    c = AALTO_COLUMNS
    rows = rows.sort(c["press_ms"])
    if rows.is_empty():
        return []

    events: list[Event] = []
    t0: int | None = None

    for row in rows.iter_rows(named=True):
        press_raw, release_raw = row[c["press_ms"]], row[c["release_ms"]]
        if press_raw is None:
            continue
        if t0 is None:
            t0 = int(press_raw)

        press = int(press_raw) - t0
        release = int(release_raw) - t0 if release_raw is not None else press
        release = max(release, press)  # guard against logging jitter

        letter = row[c["char"]]
        if letter is None:
            continue
        letter = str(letter)
        if letter.upper() in _BACKSPACE_MARKERS:
            events.append(Event.backspace(press, release))
        elif len(letter) == 1:
            events.append(Event.key(letter, press, release))
        # Multi-char names (SHIFT, CAPS_LOCK, ...) are modifier rows: skip.

    # Anchor on the first keystroke, not the first row: sessions often open
    # with a SHIFT or a click that produces no event.
    return rebase(events)


def _has_unlogged_letters(rows: pl.DataFrame) -> bool:
    """True if any row lost its key identity.

    The readme notes the logger sometimes records only a JavaScript keycode.
    Keycodes cannot recover letter case, so rather than guess we drop the
    session — 90% of participants are unaffected and the corpus is huge.
    """
    return rows[AALTO_COLUMNS["char"]].null_count() > 0


def iter_sessions(path: Path) -> Iterator[tuple[str, str, list[Event]]]:
    """Yields (participant_id, target_text, events) per test section.

    `path` may be a single log or a directory of them.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*_keystrokes.txt"))
    else:
        files = [path]

    c = AALTO_COLUMNS
    for log_path in files:
        try:
            df = read_log(log_path)
        except Exception:  # noqa: BLE001 - a corrupt log must not stop the sweep
            continue
        if any(col not in df.columns for col in (c["participant"], c["session"])):
            continue

        for _key, group in df.group_by(
            [c["participant"], c["session"]], maintain_order=True
        ):
            if _has_unlogged_letters(group):
                continue
            events = parse_session(group)
            if not events:
                continue
            participant = str(group[c["participant"]][0])
            target = str(group[c["target"]][0])
            yield participant, target, events


def physical_keyboard_participants(metadata_path: Path) -> set[str]:
    """Participant IDs typing on a physical keyboard, per the metadata file.

    The plan scopes this project to desktop/physical keyboards, so mobile
    ('small', 'on-screen') participants are excluded at dataset-build time.
    """
    meta = pl.read_csv(
        metadata_path,
        separator="\t",
        infer_schema_length=10_000,
        ignore_errors=True,
        encoding="utf8-lossy",
        quote_char=None,
        truncate_ragged_lines=True,
    )
    keep = meta.filter(pl.col("KEYBOARD_TYPE").is_in(list(PHYSICAL_KEYBOARDS)))
    return {str(p) for p in keep["PARTICIPANT_ID"].to_list()}
