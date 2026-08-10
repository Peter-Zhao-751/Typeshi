from pathlib import Path

import polars as pl
import pytest

from typeshi import config
from typeshi.adapters.klicke import KLICKE_COLUMNS, iter_sessions
from typeshi.buffer import replay
from typeshi.events import EventType
from typeshi.serialize import deserialize, serialize

FIXTURE = Path(__file__).parent / "fixtures" / "klicke_sample.csv"


def test_fixture_has_the_columns_the_adapter_expects():
    df = pl.read_csv(FIXTURE, n_rows=1, encoding="utf8-lossy")
    missing = [c for c in KLICKE_COLUMNS.values() if c not in df.columns]
    assert not missing, f"missing {missing}; actual columns are {df.columns}"


def test_session_contains_nonlinear_editing():
    _, _, events = next(iter_sessions(FIXTURE))
    types = {e.type for e in events}
    assert EventType.CURSOR in types or EventType.SELDEL in types


def test_final_text_matches_the_corpus_essay_on_disk():
    """The milestone is only meaningful if it is checked against the corpus's
    own essay file rather than against replay's own output.

    Trailing newlines are excluded: writers end with blank lines and the text
    exporter appends one more, so the two disagree there and nowhere else.
    """
    _, final_text, _ = next(iter_sessions(FIXTURE))
    gold = FIXTURE.with_suffix(".txt").read_text(encoding="utf-8")
    assert final_text.rstrip("\n") == gold.rstrip("\n")
    assert len(final_text.rstrip("\n")) > 100, "guard against a vacuous match"


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


def test_writer_id_comes_from_the_filename():
    """KLiCKe has no participant column; the writer is the file stem."""
    writer, _, _ = next(iter_sessions(FIXTURE))
    assert writer == FIXTURE.stem


def test_long_thinking_pauses_survive_binning():
    """Composition has multi-second pauses; binning must not flatten them.

    Gaps are preserved only up to config.MAX_MS. Real sessions contain pauses
    of several minutes (the fixture's longest is ~8), and those clamp to the
    top bin by design: spending bin resolution past two minutes would buy
    nothing, since 'walked away' is one behaviour however long it lasts.
    """
    _, _, events = next(iter_sessions(FIXTURE))
    gaps = [b.press_time - a.press_time for a, b in zip(events, events[1:])]
    restored = deserialize(serialize(events))
    rgaps = [b.press_time - a.press_time for a, b in zip(restored, restored[1:])]
    assert max(rgaps) > 0.85 * min(max(gaps), config.MAX_MS)


def test_pauses_beyond_the_ceiling_clamp_to_the_top_bin():
    """Anything past MAX_MS is indistinguishable, and that is intentional."""
    _, _, events = next(iter_sessions(FIXTURE))
    gaps = [b.press_time - a.press_time for a, b in zip(events, events[1:])]
    assert max(gaps) > config.MAX_MS, "fixture should contain an over-ceiling pause"
    restored = deserialize(serialize(events))
    rgaps = [b.press_time - a.press_time for a, b in zip(restored, restored[1:])]
    assert max(rgaps) <= config.MAX_MS


def test_sessions_that_do_not_replay_exactly_are_dropped(tmp_path):
    """~4.5% of the corpus has rollover-corrupted cursor positions. Those are
    dropped rather than patched, so exact replay stays the gate."""
    rows = pl.read_csv(FIXTURE, encoding="utf8-lossy")
    # Corrupt one cursor position so the session can no longer match the gold text.
    corrupted = rows.with_columns(
        pl.when(pl.col("").eq(50))
        .then(pl.col("CursorPosition") + 7)
        .otherwise(pl.col("CursorPosition"))
        .alias("CursorPosition")
    )
    bad = tmp_path / "klicke_sample.csv"
    corrupted.write_csv(bad)
    (tmp_path / "klicke_sample.txt").write_text(
        FIXTURE.with_suffix(".txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert list(iter_sessions(bad)) == []


def _klicke_rows(rows):
    """Builds a minimal KLiCKe frame from (DownTime, UpTime, DownEvent,
    CursorPosition, TextChange, Activity) tuples."""
    return pl.DataFrame({
        "DownTime": [r[0] for r in rows], "UpTime": [r[1] for r in rows],
        "DownEvent": [r[2] for r in rows], "CursorPosition": [r[3] for r in rows],
        "TextChange": [r[4] for r in rows], "Activity": [r[5] for r in rows],
    })


def test_cursor_moves_keep_their_own_timestamp():
    """A cursor jump at second 1 followed by typing at second 10 must place
    the nine-second pause AFTER the move, not before it. The lazy-seek
    version stamped the move at second 10 -- 85,970 affected rows in the
    first 500 corpus logs."""
    from typeshi.adapters.klicke import parse_session
    from typeshi.events import EventType

    text, events = parse_session(_klicke_rows([
        (0, 100, "a", 1, "a", "Input"),
        (200, 300, "b", 2, "b", "Input"),
        (1000, 1000, "Leftclick", 0, "NoChange", "Nonproduction"),
        (10000, 10100, "c", 1, "c", "Input"),
    ]))
    assert text == "cab"
    cursor = next(e for e in events if e.type is EventType.CURSOR)
    assert cursor.press_time == 1000, (
        f"cursor move stamped at {cursor.press_time}, expected its own 1000"
    )


def test_paste_sessions_are_dropped_not_flattened():
    """A 100-char paste is not 100 keystrokes at one timestamp; sessions
    containing Paste (13.8% of the corpus) or Move are dropped."""
    from typeshi.adapters.klicke import UnsupportedSession, parse_session

    with pytest.raises(UnsupportedSession):
        parse_session(_klicke_rows([
            (0, 100, "a", 1, "a", "Input"),
            (500, 500, "v", 13, "pasted text!", "Paste"),
        ]))
