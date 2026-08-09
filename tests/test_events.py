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
