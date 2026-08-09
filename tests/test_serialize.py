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
