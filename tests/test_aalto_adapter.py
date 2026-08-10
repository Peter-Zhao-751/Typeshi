from pathlib import Path

import polars as pl
import pytest

from typeshi.adapters.aalto import AALTO_COLUMNS, iter_sessions, parse_session, read_log
from typeshi.buffer import replay
from typeshi.events import EventType
from typeshi.labels import _levenshtein

FIXTURE = Path(__file__).parent / "fixtures" / "aalto_sample.txt"


def test_fixture_has_the_columns_the_adapter_expects():
    """Fails loudly with the real header if the corpus schema differs."""
    df = pl.read_csv(FIXTURE, separator="\t", n_rows=1, quote_char=None)
    missing = [c for c in AALTO_COLUMNS.values() if c not in df.columns]
    assert not missing, f"missing {missing}; actual columns are {df.columns}"


def test_parses_events_in_press_time_order():
    _, _, events = next(iter_sessions(FIXTURE))
    times = [e.press_time for e in events]
    assert times == sorted(times)


def test_only_key_and_backspace_events_are_emitted():
    """Aalto is transcription: linear typing, no cursor jumps."""
    for _, _, events in iter_sessions(FIXTURE):
        assert {e.type for e in events} <= {EventType.KEY, EventType.BACKSPACE}


def test_press_times_are_zero_based_ms_ints():
    _, _, events = next(iter_sessions(FIXTURE))
    assert events[0].press_time == 0
    assert all(isinstance(e.press_time, int) for e in events)


def test_epoch_timestamps_are_rebased_not_left_absolute():
    """Raw PRESS_TIME is Unix epoch ms; sessions must be zero-based."""
    raw = read_log(FIXTURE)
    assert raw[AALTO_COLUMNS["press_ms"]].min() > 1_000_000_000_000
    for _, _, events in iter_sessions(FIXTURE):
        assert max(e.press_time for e in events) < 10_000_000


def test_hold_times_are_non_negative():
    for _, _, events in iter_sessions(FIXTURE):
        assert all(e.release_time >= e.press_time for e in events)


def test_modifier_rows_are_skipped():
    """SHIFT and CAPS_LOCK rows carry no character and must not become keys."""
    _, _, events = next(iter_sessions(FIXTURE))
    assert all(e.char != "SHIFT" for e in events)
    assert all(e.char is None or len(e.char) == 1 for e in events)


def test_backspace_rows_become_backspace_events():
    types = {e.type for _, _, ev in iter_sessions(FIXTURE) for e in ev}
    assert EventType.BACKSPACE in types


def test_replayed_text_matches_what_the_participant_actually_typed():
    """USER_INPUT records the text the participant submitted, a far tighter
    check than the target sentence: transcription may legitimately diverge
    from SENTENCE, but replay should track what the log says was produced.

    Similarity is edit distance, not positional overlap: a single dropped
    keystroke shifts every later character and would tank a positional score
    while the text is plainly still correct. Measured over 1,605 real
    sessions, 76% replay USER_INPUT exactly and 98% score at least 0.90.
    """
    log = read_log(FIXTURE)
    c = AALTO_COLUMNS
    checked = 0
    for session_id in log[c["session"]].unique():
        rows = log.filter(pl.col(c["session"]) == session_id)
        submitted = str(rows[c["user_input"]][0])
        produced = replay(parse_session(rows))
        similarity = 1 - _levenshtein(produced, submitted) / max(len(submitted), 1)
        assert similarity > 0.9, f"{produced!r} does not track {submitted!r}"
        checked += 1
    assert checked >= 4


def test_replayed_text_is_close_to_the_recorded_target():
    """Transcription is imperfect, so allow small divergence but not garbage."""
    _, target, events = next(iter_sessions(FIXTURE))
    produced = replay(events)
    overlap = sum(a == b for a, b in zip(produced, target))
    assert overlap / max(len(target), 1) > 0.5


def test_multiple_sessions_are_yielded():
    """One participant file holds 15 sentences; the fixture spans two people."""
    sessions = list(iter_sessions(FIXTURE))
    assert len(sessions) >= 2
    assert len({p for p, _, _ in sessions}) == 2


def test_sessions_with_unlogged_letters_are_dropped(tmp_path):
    """~8% of rows have a null LETTER (the readme's 'keycode used instead'
    case). Case cannot be recovered reliably, so those sessions are dropped."""
    rows = read_log(FIXTURE)
    first = rows[AALTO_COLUMNS["session"]][0]
    damaged = rows.with_columns(
        pl.when(pl.col(AALTO_COLUMNS["session"]).eq(first))
        .then(None)
        .otherwise(pl.col(AALTO_COLUMNS["char"]))
        .alias(AALTO_COLUMNS["char"])
    )
    path = tmp_path / "damaged.txt"
    damaged.write_csv(path, separator="\t")

    before = len(list(iter_sessions(FIXTURE)))
    after = len(list(iter_sessions(path)))
    assert after == before - 1


def test_sessions_disagreeing_with_user_input_are_dropped(tmp_path):
    """52 of 2,745 corpus sessions replay below 0.90 similarity to the text
    the participant actually submitted (worst 0.176). They used to enter
    training silently; integrity failures are dropped."""
    rows = read_log(FIXTURE)
    first = rows[AALTO_COLUMNS["session"]][0]
    # Claim the participant submitted something the keystrokes cannot produce.
    damaged = rows.with_columns(
        pl.when(pl.col(AALTO_COLUMNS["session"]).eq(first))
        .then(pl.lit("zzz completely unrelated text zzz"))
        .otherwise(pl.col(AALTO_COLUMNS["user_input"]))
        .alias(AALTO_COLUMNS["user_input"])
    )
    path = tmp_path / "damaged.txt"
    damaged.write_csv(path, separator="\t")
    before = len(list(iter_sessions(FIXTURE)))
    after = len(list(iter_sessions(path)))
    assert after == before - 1
