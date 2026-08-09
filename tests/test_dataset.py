import pytest
from typeshi.dataset import build_examples, build_prompt, split_by_writer
from typeshi.events import Event
from typeshi.labels import SessionLabels

LABELS = SessionLabels(60.0, 0.02, 0.0, 0.05)


def _type(text, gap=100):
    return [Event.key(c, i * gap, i * gap + 50) for i, c in enumerate(text)]


def test_short_session_becomes_one_example():
    ex = build_examples("hello", _type("hello"), LABELS, mode="transcription")
    assert len(ex) == 1
    assert set(ex[0]) == {"prompt", "completion"}


def test_prompt_contains_target_text_and_knobs():
    ex = build_examples("hello", _type("hello"), LABELS, mode="transcription")
    assert "hello" in ex[0]["prompt"]
    assert "WPM=60" in ex[0]["prompt"]
    assert "MODE=transcription" in ex[0]["prompt"]


def test_completion_is_the_event_token_stream():
    ex = build_examples("hi", _type("hi"), LABELS, mode="transcription")
    assert "<KEY:h>" in ex[0]["completion"]
    assert "<DT:" in ex[0]["completion"]


def test_long_session_is_split_into_windows():
    events = _type("a" * 1200)
    ex = build_examples("a" * 1200, events, LABELS, mode="composition", max_events=512)
    assert len(ex) == 3


def test_continuation_windows_carry_resume_state():
    events = _type("a" * 1200)
    ex = build_examples("a" * 1200, events, LABELS, mode="composition", max_events=512)
    assert "CURSOR=" in ex[1]["prompt"]
    assert "CURSOR=" not in ex[0]["prompt"]


def test_windows_cover_every_event_exactly_once():
    events = _type("a" * 1000)
    ex = build_examples("a" * 1000, events, LABELS, mode="composition", max_events=512)
    total_keys = sum(e["completion"].count("<KEY:") for e in ex)
    assert total_keys == 1000


def test_resume_state_reflects_the_buffer_at_the_window_boundary():
    """A continuation prompt must describe the text as it stood when that
    window began, otherwise the model resumes from the wrong state."""
    events = _type("abcdefghij" * 60)  # 600 events
    ex = build_examples(
        "abcdefghij" * 60, events, LABELS, mode="composition", max_events=512
    )
    assert len(ex) == 2
    assert "CURSOR=512" in ex[1]["prompt"]


def test_split_is_by_writer_and_deterministic():
    writers = [f"w{i}" for i in range(100)]
    train_a, test_a = split_by_writer(writers, test_frac=0.1, seed=0)
    train_b, test_b = split_by_writer(writers, test_frac=0.1, seed=0)
    assert (train_a, test_a) == (train_b, test_b)
    assert not (train_a & test_a)
    assert len(test_a) == 10


def test_split_changes_with_seed():
    writers = [f"w{i}" for i in range(100)]
    _, test_a = split_by_writer(writers, test_frac=0.1, seed=0)
    _, test_b = split_by_writer(writers, test_frac=0.1, seed=1)
    assert test_a != test_b


def test_split_ignores_duplicate_writer_ids():
    """Sessions repeat a writer many times; the split works on unique writers."""
    writers = [f"w{i % 20}" for i in range(500)]
    train, test = split_by_writer(writers, test_frac=0.1, seed=0)
    assert len(train | test) == 20
    assert not (train & test)


def test_build_prompt_is_shared_by_training_and_inference():
    """Training export and generate() must produce byte-identical prompts."""
    from_export = build_examples("hi", _type("hi"), LABELS, mode="transcription")[0]
    direct = build_prompt("hi", LABELS, "transcription")
    assert from_export["prompt"] == direct


def test_empty_session_produces_no_examples():
    assert build_examples("hello", [], LABELS, mode="transcription") == []
