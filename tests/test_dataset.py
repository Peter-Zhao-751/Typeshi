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


def test_prompt_contains_target_text_and_knob_tokens():
    ex = build_examples("hello", _type("hello"), LABELS, mode="transcription")
    assert "<TARGET>hello" in ex[0]["prompt"]
    assert "<WPM:12>" in ex[0]["prompt"]          # 60 wpm // 5
    assert ex[0]["prompt"].startswith("<MODE:T>")
    assert ex[0]["prompt"].endswith("<PROCESS>")


def test_prompt_has_no_instruction_boilerplate():
    """The v1 preamble was 10 constant tokens in 2M examples; it is gone."""
    ex = build_examples("hello", _type("hello"), LABELS, mode="transcription")
    assert "Simulate" not in ex[0]["prompt"]


def test_completion_is_the_event_token_stream():
    ex = build_examples("hi", _type("hi"), LABELS, mode="transcription")
    assert "<h:" in ex[0]["completion"]
    assert "<DT:" in ex[0]["completion"]
    assert not ex[0]["completion"].startswith("<DT:")


def test_long_session_is_split_into_windows():
    events = _type("a" * 1200)
    ex = build_examples("a" * 1200, events, LABELS, mode="composition", max_events=512)
    assert len(ex) == 3


def test_continuation_windows_carry_resume_state():
    events = _type("a" * 1200)
    ex = build_examples("a" * 1200, events, LABELS, mode="composition", max_events=512)
    assert "<WRITTEN>" in ex[1]["prompt"] and "<CUR:512>" in ex[1]["prompt"]
    assert "<WRITTEN>" not in ex[0]["prompt"]


def test_windows_cover_every_event_exactly_once():
    events = _type("a" * 1000)
    ex = build_examples("a" * 1000, events, LABELS, mode="composition", max_events=512)
    total_keys = sum(e["completion"].count("<a:") for e in ex)
    assert total_keys == 1000


def test_resume_state_reflects_the_buffer_at_the_window_boundary():
    """A continuation prompt must describe the text as it stood when that
    window began, otherwise the model resumes from the wrong state."""
    events = _type("abcdefghij" * 60)  # 600 events
    ex = build_examples(
        "abcdefghij" * 60, events, LABELS, mode="composition", max_events=512
    )
    assert len(ex) == 2
    assert "<CUR:512>" in ex[1]["prompt"]


def test_window_boundary_gap_is_preserved_not_zeroed():
    """v1 serialized each window independently, so the gap between the last
    event of one window and the first of the next was silently recorded as
    zero -- in composition that can be a minutes-long pause."""
    events = _type("a" * 600, gap=100)
    ex = build_examples("a" * 600, events, LABELS, mode="composition", max_events=512)
    assert ex[1]["completion"].startswith("<DT:")
    from typeshi.serialize import deserialize
    restored = deserialize(ex[1]["completion"])
    assert abs(restored[0].press_time - 100) <= 15  # the real 100ms boundary gap


def test_split_is_by_writer_and_deterministic():
    writers = [f"w{i}" for i in range(100)]
    train_a, test_a = split_by_writer(writers, test_frac=0.1, seed=0)
    train_b, test_b = split_by_writer(writers, test_frac=0.1, seed=0)
    assert (train_a, test_a) == (train_b, test_b)
    assert not (train_a & test_a)
    assert train_a | test_a == set(writers)
    # Writers are assigned independently so the split stays stable under
    # subsetting, which makes the test fraction statistical, not exact. On 100
    # writers that is a loose band; at corpus scale it converges tightly.
    assert 3 <= len(test_a) <= 20


def test_split_fraction_converges_at_scale():
    """The 3..20 band above is loose by necessity (100 writers); at corpus
    scale the independent assignments must actually deliver test_frac, or a
    2x-off holdout could hide behind the small-n tolerance forever."""
    writers = [f"w{i}" for i in range(10_000)]
    _, test = split_by_writer(writers, test_frac=0.1, seed=0)
    assert 0.09 <= len(test) / len(writers) <= 0.11


def test_split_is_stable_under_subsetting():
    """A subset build and a full build must agree on every shared writer.

    The corpus is built at several sizes via --limit-aalto. If the holdout
    depended on which writers were present, a writer held out of the subset run
    could sit in the full run's training set, and the two checkpoints' Tier-1
    numbers would not be comparable.
    """
    full = [f"w{i}" for i in range(1000)]
    subset = [f"w{i}" for i in range(0, 1000, 3)]

    _, full_test = split_by_writer(full, test_frac=0.1, seed=0)
    subset_train, subset_test = split_by_writer(subset, test_frac=0.1, seed=0)

    for writer in subset:
        assert (writer in subset_test) == (writer in full_test)
    assert subset_test  # a vacuous split would satisfy the loop above
    assert subset_train


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


def test_written_state_is_capped_to_a_tail():
    """Uncapped WRITTEN state pushed 10 of 4,680 sample prompts past the
    trainer's max_length, truncating their completions."""
    from typeshi.dataset import WRITTEN_TAIL_CHARS, build_prompt

    long_text = "x" * 3000
    prompt = build_prompt("target", LABELS, "composition",
                          written_so_far=long_text, cursor=3000)
    written = prompt.split("<WRITTEN>")[1].split("<CUR:")[0]
    assert len(written) == WRITTEN_TAIL_CHARS


def test_corpus_text_containing_a_marker_is_refused():
    from typeshi.dataset import build_prompt

    with pytest.raises(ValueError):
        build_prompt("essay quoting <PROCESS> literally", LABELS, "composition")


def test_unsupported_chars_flags_non_ascii_keys():
    from typeshi.serialize import unsupported_chars

    ok = _type("hello world")
    assert unsupported_chars(ok) == set()
    cyrillic = ok + [Event.key("е", 5000, 5050)]
    assert unsupported_chars(cyrillic) == {"е"}
