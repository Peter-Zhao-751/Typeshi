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
