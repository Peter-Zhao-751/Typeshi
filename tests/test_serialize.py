import pytest
from typeshi import config
from typeshi.events import Event, EventType
from typeshi.serialize import (
    PCT_MAX,
    WPM_BINS,
    deserialize,
    pct_bin,
    serialize,
    special_tokens,
    wpm_bin,
)


def test_one_keystroke_is_two_tokens_and_no_leading_dt():
    s = serialize([Event.key("h", 100, 160), Event.key("i", 220, 270)])
    assert not s.startswith("<DT:"), "leading <DT:0> was constant noise, dropped in v2"
    assert s.count("<h:") == 1 and s.count("<i:") == 1
    assert s.count("<DT:") == 1  # only between the two events


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
    hold_in = events[0].release_time - events[0].press_time
    hold_out = out[0].release_time - out[0].press_time
    assert abs(hold_out - hold_in) / hold_in < 0.15


def test_special_characters_are_escaped():
    events = [Event.key(" ", 0, 50), Event.key("\n", 100, 150),
              Event.key("<", 200, 250), Event.key(">", 300, 350)]
    s = serialize(events)
    assert "<SPC:" in s and "<NL:" in s and "<LT:" in s and "<GT:" in s
    out = deserialize(s)
    assert [e.char for e in out] == [" ", "\n", "<", ">"]


def test_colon_as_a_typed_character_survives():
    out = deserialize(serialize([Event.key(":", 0, 60)]))
    assert out[0].char == ":"


def test_rollover_reads_as_hold_bin_greater_than_dt_bin():
    """<X:y><DT:z> with y > z means X was still down at the next press.
    Valid only because holds and gaps share one bin scale."""
    events = [Event.key("t", 0, 120), Event.key("h", 60, 180)]
    s = serialize(events)
    import re
    hold_bin = int(re.match(r"<t:(\d+)>", s).group(1))
    dt_bin = int(re.search(r"<DT:(\d+)>", s).group(1))
    assert hold_bin > dt_bin

    out = deserialize(s)
    assert out[0].release_time > out[1].press_time  # still overlapping after decode


def test_continuation_window_carries_the_boundary_gap():
    e = Event.key("a", 5000, 5050)
    s = serialize([e], prev_press_time=0)
    assert s.startswith("<DT:")
    out = deserialize(s)
    assert abs(out[0].press_time - 5000) / 5000 < 0.15


def test_deserialize_rejects_malformed_input():
    with pytest.raises(ValueError):
        deserialize("<a:1>junk")
    with pytest.raises(ValueError):
        deserialize("<NONSENSE:1>")


def test_deserialize_rejects_missing_dt_between_events():
    with pytest.raises(ValueError):
        deserialize("<a:1><b:2>")


def test_deserialize_rejects_double_and_dangling_dt():
    with pytest.raises(ValueError):
        deserialize("<DT:1><DT:2><a:1>")
    with pytest.raises(ValueError):
        deserialize("<a:1><DT:5>")


def test_empty_stream_round_trips():
    assert serialize([]) == ""
    assert deserialize("") == []


def test_knob_binning_clamps_and_scales():
    assert wpm_bin(61.4) == 12
    assert wpm_bin(92) == 18
    assert wpm_bin(9999) == WPM_BINS - 1
    assert pct_bin(0.031) == 3
    assert pct_bin(0.004) == 0
    assert pct_bin(1.0) == PCT_MAX


def test_special_tokens_cover_grammar():
    toks = set(special_tokens())
    assert {"<DT:0>", "<DT:127>", "<a:0>", "<a:127>", "<SPC:51>",
            "<BKSP:0>", "<MODE:T>", "<MODE:C>", f"<WPM:{WPM_BINS - 1}>",
            f"<ECOR:{PCT_MAX}>", "<TARGET>", "<WRITTEN>", "<PROCESS>"} <= toks
    # CUR/SELDEL are regex-parsed prefixes, not whole tokens
    assert "<CUR:" in toks and "<SELDEL:" in toks
    whole = [t for t in toks if t.endswith(">")]
    assert len(whole) == len(toks) - 2


def test_vocabulary_size_is_as_designed():
    """97 char identities (92 direct printable + 5 named escapes) x 128 holds,
    plus BKSP x 128, DT x 128, knobs, and markers."""
    whole = [t for t in special_tokens() if t.endswith(">")]
    expected = 97 * 128 + 128 + 128 + 2 + WPM_BINS + 3 * (PCT_MAX + 1) + 3
    assert len(whole) == expected
    assert len(whole) == 12_810
