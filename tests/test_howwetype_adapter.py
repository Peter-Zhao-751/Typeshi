from pathlib import Path

import polars as pl
import pytest

from typeshi.adapters.howwetype import (
    FINGER_LABELS,
    HOWWETYPE_COLUMNS,
    iter_sessions,
    iter_sessions_with_fingers,
    read_log,
)
from typeshi.buffer import replay
from typeshi.events import EventType

FIXTURE = Path(__file__).parent / "fixtures" / "howwetype_sample_matched.txt"


def test_fixture_has_the_columns_the_adapter_expects():
    """Fails loudly with the real header if the corpus schema differs."""
    df = pl.read_csv(FIXTURE, separator="\t", n_rows=1)
    missing = [c for c in HOWWETYPE_COLUMNS.values() if c not in df.columns]
    assert not missing, f"missing {missing}; actual columns are {df.columns}"


def test_parses_events_in_press_time_order():
    _, _, events = next(iter_sessions(FIXTURE))
    times = [e.press_time for e in events]
    assert times == sorted(times)


def test_only_key_and_backspace_events_are_emitted():
    """How We Type is transcription: linear typing, no cursor jumps."""
    for _, _, events in iter_sessions(FIXTURE):
        assert {e.type for e in events} <= {EventType.KEY, EventType.BACKSPACE}


def test_press_times_are_zero_based_ms_ints():
    _, _, events = next(iter_sessions(FIXTURE))
    assert events[0].press_time == 0
    assert all(isinstance(e.press_time, int) for e in events)


def test_epoch_seconds_are_converted_to_rebased_ms():
    """Raw input_time is Unix epoch *seconds* (float); events must be ms."""
    raw = read_log(FIXTURE)
    assert raw[HOWWETYPE_COLUMNS["press_s"]].min() > 1_000_000_000  # epoch s
    for _, _, events in iter_sessions(FIXTURE):
        assert max(e.press_time for e in events) < 10_000_000  # rebased ms


def test_release_equals_press_because_corpus_has_no_key_ups():
    """The corpus logs key-downs only; hold times cannot come from it."""
    for _, _, events in iter_sessions(FIXTURE):
        assert all(e.release_time == e.press_time for e in events)


def test_modifier_rows_are_skipped():
    """Shift_L / Shift_R rows must not become events of their own."""
    for _, _, events in iter_sessions(FIXTURE):
        assert all(e.char is None or len(e.char) == 1 for e in events)


def test_backspace_rows_become_backspace_events():
    types = {e.type for _, _, ev in iter_sessions(FIXTURE) for e in ev}
    assert EventType.BACKSPACE in types


def test_space_keysym_becomes_a_space_not_an_underscore():
    """The `input` column renders space as `_`; the event must carry ' '."""
    chars = {e.char for _, _, ev in iter_sessions(FIXTURE) for e in ev if e.char}
    assert " " in chars
    assert "_" not in chars


def test_shift_capitalization_replays_the_submitted_text_exactly():
    """Letters are logged lowercase even when shifted; the adapter arms
    capitalization off the preceding Shift row. Every fixture block was cut
    from sessions that replay `current_input` byte-exact, so any drift here
    is an adapter regression, not corpus noise."""
    raw = read_log(FIXTURE)
    c = HOWWETYPE_COLUMNS
    submitted = {
        (str(k[0]), str(k[1])): str(g[c["user_input"]][0])
        for k, g in raw.group_by([c["participant"], c["session"]])
    }
    checked = 0
    for participant, _, events in iter_sessions(FIXTURE):
        text = replay(events)
        assert text[0].isupper()
        assert text in submitted.values()
        checked += 1
    assert checked >= 4


def test_finger_annotations_align_one_to_one_with_events():
    """The point of this corpus: every keypress carries the motion-capture
    finger label, giving supervised ground truth for same-finger detection."""
    for _, _, events, fingers in iter_sessions_with_fingers(FIXTURE):
        assert len(fingers) == len(events)
        assert all(f in FINGER_LABELS for f in fingers)


def test_backspace_events_carry_finger_labels_too():
    labelled = [
        f
        for _, _, ev, fg in iter_sessions_with_fingers(FIXTURE)
        for e, f in zip(ev, fg)
        if e.type == EventType.BACKSPACE
    ]
    assert labelled and all(f in FINGER_LABELS for f in labelled)


def test_finger_sequence_matches_the_recorded_annotations():
    """Pins the exact event/finger alignment for one real block (read off the
    corpus bytes). Participant 900002 presses space with the left thumb but
    'n' with the left index — idiosyncratic strategies are exactly what the
    annotations are for, so the adapter must not reorder or shift them."""
    block = next(
        (ev, fg)
        for p, t, ev, fg in iter_sessions_with_fingers(FIXTURE)
        if p == "900002" and t == "Onko se ohi?"
    )
    events, fingers = block
    assert [e.char for e in events] == list("Onko se ohi?")
    assert fingers == [
        "R_Index", "L_Index", "R_Index", "R_Middle", "L_Thumb", "L_Ring",
        "L_Middle", "L_Thumb", "R_Middle", "L_Index", "R_Index", "R_Index",
    ]


def test_iter_sessions_is_the_finger_stream_minus_annotations():
    plain = list(iter_sessions(FIXTURE))
    full = list(iter_sessions_with_fingers(FIXTURE))
    assert [(p, t, ev) for p, t, ev, _ in full] == plain


def test_multiple_sessions_and_participants_are_yielded():
    """Real corpus files are one participant each; the fixture packs blocks
    from two (anonymized) participants, as the Aalto fixture does."""
    sessions = list(iter_sessions(FIXTURE))
    assert len(sessions) >= 4
    assert len({p for p, _, _ in sessions}) == 2


def test_sessions_disagreeing_with_current_input_are_dropped(tmp_path):
    """`current_input` is the submitted text and the replay gate. A block
    whose rows cannot produce it (rollover-corrupted logs, 2 of 1,499 in the
    corpus) must be dropped, not patched."""
    rows = read_log(FIXTURE)
    c = HOWWETYPE_COLUMNS
    first = rows[c["session"]][0]
    first_participant = rows[c["participant"]][0]
    damaged = rows.with_columns(
        pl.when(
            pl.col(c["session"]).eq(first)
            & pl.col(c["participant"]).eq(first_participant)
        )
        .then(pl.lit("zzz completely unrelated text zzz"))
        .otherwise(pl.col(c["user_input"]))
        .alias(c["user_input"])
    )
    path = tmp_path / "damaged_matched.txt"
    damaged.write_csv(path, separator="\t")

    before = len(list(iter_sessions(FIXTURE)))
    after = len(list(iter_sessions(path)))
    assert after == before - 1


def test_phantom_unnamed_column_is_tolerated(tmp_path):
    """One corpus file (549687) carries an extra `Unnamed: 10` column, so
    columns must be selected by name, never by position."""
    rows = pl.read_csv(FIXTURE, separator="\t")
    with_phantom = rows.insert_column(
        10, pl.Series("Unnamed: 10", [None] * len(rows))
    )
    path = tmp_path / "phantom_matched.txt"
    with_phantom.write_csv(path, separator="\t")

    assert len(list(iter_sessions(path))) == len(list(iter_sessions(FIXTURE)))
