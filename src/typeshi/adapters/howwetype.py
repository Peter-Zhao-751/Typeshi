"""How We Type (Feit, Weir & Oulasvirta, CHI 2016) -> canonical events.

Transcription with per-keystroke *finger annotations* from motion capture —
the point of this corpus: it turns same-finger detection into supervised
ground truth for the keyboard-reconstruction workstream.

License: **CC BY-NC 4.0** (Zenodo record 4034268). Non-commercial research
use with attribution; cite Feit, Weir & Oulasvirta, *How We Type: Movement
Strategies and Performance in Everyday Typing*, CHI 2016
(doi 10.1145/2858036.2858233).

Schema notes that drive this file (see docs/data-schemas.md, "How We Type"):
  - TAB-separated `<user>_log_Sentences_<epoch>_matched.txt`, one per person
  - `input_time` is Unix epoch *seconds* (float, ms decimals) — x1000, rebase
  - key-down only: **there are no release times in this corpus**
  - letters are logged lowercase even when shifted; case is reconstructed by
    capitalizing the next character-producing keypress after a Shift row
  - read WITH the default quote char (files carry CSV quoting), the opposite
    of the Aalto reader
  - `finger` labels every keypress row: `{L,R}_{Thumb,Index,Middle,Ring,Little}`
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl

from typeshi.buffer import replay
from typeshi.events import Event, rebase
from typeshi.labels import _levenshtein

# Canonical name -> real column. Verified against the corpus header (which
# disagrees with the bundled Readme.txt — see docs/data-schemas.md).
HOWWETYPE_COLUMNS = {
    "participant": "user_id",
    "session": "stimulus_id",
    "target": "stimulus",
    "user_input": "current_input",
    "press_s": "input_time",
    "char": "input",
    "key": "key_symbol",
    "finger": "finger",
}

# The ten annotation values observed in the release (zero nulls corpus-wide).
FINGER_LABELS = frozenset(
    f"{hand}_{finger}"
    for hand in ("L", "R")
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little")
)

# Modifier keysyms: no text effect, never become events. Shifts additionally
# arm capitalization of the next character-producing keypress.
_SHIFT = {"Shift_L", "Shift_R"}
_MODIFIERS = _SHIFT | {
    "Control_L",
    "Control_R",
    "Alt_L",
    "Alt_R",
    "Multi_key",
    "Caps_Lock",
    "ISO_Level3_Shift",
}
_BACKSPACE = "BackSpace"
_SPACE = "space"

# Same gate as the Aalto adapter: replayed events must reproduce the text the
# participant actually submitted (`current_input`). Measured over all 1,499
# blocks: 98.1% replay byte-exact, 99.87% score >= 0.90; the 2 failures are
# rollover-corrupted logs (stale row order), the same shape as the other
# corpora. Integrity failures are dropped, not patched.
REPLAY_SIMILARITY_MIN = 0.90


def read_log(path: Path) -> pl.DataFrame:
    """Reads one typing log.

    Unlike the Aalto reader this keeps the default quote char: these files
    were written with CSV quoting, so a field containing a quote mark arrives
    wrapped in quotes with the inner quote doubled unless quotes are
    honoured. Columns are selected by name because one corpus file carries a
    phantom `Unnamed: 10` column.
    """
    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10_000,
        ignore_errors=True,
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
        schema_overrides={
            HOWWETYPE_COLUMNS["char"]: pl.Utf8,
            HOWWETYPE_COLUMNS["key"]: pl.Utf8,
            HOWWETYPE_COLUMNS["finger"]: pl.Utf8,
        },
    )
    wanted = [c for c in HOWWETYPE_COLUMNS.values() if c in df.columns]
    return df.select(wanted)


def parse_session(rows: pl.DataFrame) -> tuple[list[Event], list[str | None]]:
    """One stimulus block -> (events, fingers), times rebased to ms from zero.

    `fingers[i]` is the motion-capture finger label for `events[i]` — this is
    the supervised ground truth accessor. The corpus records key-downs only,
    so `release_time` is set equal to `press_time`; hold times cannot come
    from this corpus.

    Case reconstruction: letter keysyms are logged lowercase even when
    shifted (shifted *symbols* arrive already resolved, e.g. `question`),
    so a Shift row arms capitalization of the next character-producing
    keypress. Dead-key rows with multi-char `input` emit one KEY per
    rendered char at the same timestamp; blocks they corrupt fail the
    replay gate downstream.
    """
    c = HOWWETYPE_COLUMNS
    rows = rows.sort(c["press_s"])
    if rows.is_empty():
        return [], []

    events: list[Event] = []
    fingers: list[str | None] = []
    shift_armed = False

    for row in rows.iter_rows(named=True):
        press_raw = row[c["press_s"]]
        key, char = row[c["key"]], row[c["char"]]
        if press_raw is None or key is None or char is None:
            continue
        press = round(float(press_raw) * 1000)  # epoch s -> epoch ms
        key = str(key)

        if key in _MODIFIERS:
            if key in _SHIFT:
                shift_armed = True
            continue

        finger = row[c["finger"]]
        finger = str(finger) if finger is not None else None

        if key == _BACKSPACE:
            events.append(Event.backspace(press, press))
            fingers.append(finger)
        else:
            text = " " if key == _SPACE else str(char)
            if shift_armed:
                text = text.upper()
            for ch in text:
                events.append(Event.key(ch, press, press))
                fingers.append(finger)
        shift_armed = False

    events = rebase(events)  # rebase preserves order, so fingers stay aligned
    return events, fingers


def iter_sessions_with_fingers(
    path: Path,
) -> Iterator[tuple[str, str, list[Event], list[str | None]]]:
    """Yields (participant, target_text, events, fingers) per stimulus block.

    `path` may be a single log or a directory of them. `fingers` is aligned
    1:1 with `events` (KEY and BACKSPACE rows both carry a label). Blocks
    whose replay does not track the submitted `current_input` are dropped,
    exactly like the Aalto adapter's gate.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*_matched.txt"))
    else:
        files = [path]

    c = HOWWETYPE_COLUMNS
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
            events, fingers = parse_session(group)
            if not events:
                continue
            submitted = group[c["user_input"]][0]
            if submitted is None:
                continue  # nothing to verify against -> cannot trust the rows
            submitted = str(submitted)
            produced = replay(events)
            similarity = 1 - _levenshtein(produced, submitted) / max(len(submitted), 1)
            if similarity < REPLAY_SIMILARITY_MIN:
                continue
            participant = str(group[c["participant"]][0])
            target = str(group[c["target"]][0])
            yield participant, target, events, fingers


def iter_sessions(path: Path) -> Iterator[tuple[str, str, list[Event]]]:
    """Yields (participant_id, target_text, events) per stimulus block.

    Same stream as `iter_sessions_with_fingers`, minus the annotations, so it
    matches the other adapters' signature.
    """
    for participant, target, events, _fingers in iter_sessions_with_fingers(path):
        yield participant, target, events
