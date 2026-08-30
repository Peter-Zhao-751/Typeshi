import pytest
from typeshi.dataset import build_examples, build_prompt, split_by_writer
from typeshi.serialize import wpm_bin
from typeshi.events import Event
from typeshi.labels import SessionLabels, window_labels

LABELS = SessionLabels(60.0, 0.02, 0.0, 0.05)


def _type(text, gap=100):
    return [Event.key(c, i * gap, i * gap + 50) for i, c in enumerate(text)]


def test_short_session_becomes_one_example():
    ex = build_examples("hello", _type("hello"), LABELS, mode="transcription")
    assert len(ex) == 1
    assert set(ex[0]) == {"prompt", "completion"}


def test_prompt_contains_target_text_and_knob_tokens():
    events = _type("hello")
    ex = build_examples("hello", events, LABELS, mode="transcription")
    assert "<TARGET>hello" in ex[0]["prompt"]
    # The speed token describes the window's OWN typing, not a caller-supplied
    # session figure -- attaching session values to windows is the
    # mislabelling that made these knobs unlearnable in composition.
    assert f"<WPM:{wpm_bin(window_labels(events, LABELS).wpm)}>" in ex[0]["prompt"]
    assert ex[0]["prompt"].startswith("<MODE:T>")
    assert ex[0]["prompt"].endswith("<PROCESS>")


def test_window_label_schedule_matches_the_labels_training_embedded():
    """Generation needs the same per-window labels training saw. The schedule
    of a real session must reproduce, window for window, exactly the knob
    header build_examples put in each window's prompt -- any drift and
    generation is conditioned on something the model was not taught."""
    from typeshi.dataset import window_label_schedule

    events = _type("a" * 700)
    # A revision burst in the second window so its labels differ from both
    # the first window's and the session aggregate.
    events[600] = Event.cursor(3, events[600].press_time)
    events[601] = Event.seldel(1, 3, events[601].press_time)
    ex = build_examples("a" * 700, events, LABELS, mode="composition",
                        max_events=512)
    schedule = window_label_schedule(events, LABELS, max_events=512)

    assert len(schedule) == len(ex) == 2
    for labels, example in zip(schedule, ex):
        assert example["prompt"].startswith(labels.to_tokens("composition"))
    assert schedule[0].revision_rate != schedule[1].revision_rate


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
    events = _type("hi")
    from_export = build_examples("hi", events, LABELS, mode="transcription")[0]
    # Same function, same bytes -- but the export derives its labels from the
    # window rather than taking them on faith.
    direct = build_prompt("hi", window_labels(events, LABELS), "transcription")
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


def test_window_labels_describe_the_window_not_the_session():
    """Session labels on every window is a mislabelling, not a rounding.

    Measured over 24,909 exported composition windows: the <REV:> bin matched
    what that window actually did only 52.9% of the time -- 39.9% of windows
    labelled <REV:0> do revise, and 24.0% labelled <REV:n>, n>0, do not. A
    third of the signal teaches the model that the token predicts nothing,
    which is exactly the behaviour the knob shows.
    """
    from typeshi.dataset import build_examples
    from typeshi.events import Event
    from typeshi.labels import compute_labels

    # Window 1: four plain keystrokes. Window 2: three keys and one cursor op.
    events = [Event.key(c, i * 100, i * 100 + 50)
              for i, c in enumerate("abcd")]
    events += [Event.key("e", 400, 450), Event.cursor(0, 500),
               Event.key("f", 600, 650), Event.key("g", 700, 750)]
    session = compute_labels(events, "abcdefg")

    examples = build_examples("abcdefg", events, session, "composition",
                              max_events=4)

    assert len(examples) == 2
    from typeshi.serialize import rev_bin

    assert "<REV:0>" in examples[0]["prompt"], "a window with no cursor op"
    assert f"<REV:{rev_bin(0.25)}>" in examples[1]["prompt"], \
        "one cursor op in four events"


def test_window_labels_inherit_the_sessions_uncorrected_rate():
    """EUNC is how far the FINAL text lands from the target -- a whole-session
    quantity. A window produces only part of the text, so recomputing it per
    window would measure a shortfall that later windows go on to fill."""
    from typeshi.dataset import build_examples
    from typeshi.events import Event
    from typeshi.labels import compute_labels
    from typeshi.serialize import pct_bin

    events = [Event.key(c, i * 100, i * 100 + 50)
              for i, c in enumerate("abcdefgh")]
    session = compute_labels(events, "abcdefgh")
    examples = build_examples("abcdefgh", events, session, "transcription",
                              max_events=4)

    want = f"<EUNC:{pct_bin(session.uncorrected_error_rate)}>"
    assert all(want in ex["prompt"] for ex in examples)


def test_revision_repeats_oversamples_only_high_revision_windows():
    """Only 0.9% of exported composition windows sit at REV >= 5, so the
    behaviour is present but far too rare to learn. Oversampling is only
    meaningful once the labels are per-window -- before that it would have
    duplicated whatever the session average happened to say."""
    from typeshi.dataset import revision_repeats

    high = "<MODE:C><WPM:8><ECOR:12><EUNC:0><REV:9><TARGET>x<PROCESS>"
    low = "<MODE:C><WPM:8><ECOR:12><EUNC:0><REV:1><TARGET>x<PROCESS>"
    assert revision_repeats(high, factor=20, min_bin=5) == 20
    assert revision_repeats(low, factor=20, min_bin=5) == 1


def test_revision_repeats_is_inert_by_default():
    """The export must be byte-identical unless oversampling is asked for."""
    from typeshi.dataset import revision_repeats

    prompt = "<MODE:C><WPM:8><ECOR:12><EUNC:0><REV:9><TARGET>x<PROCESS>"
    assert revision_repeats(prompt, factor=1, min_bin=5) == 1
    assert revision_repeats("no rev token here", factor=20, min_bin=5) == 1


def test_single_window_labels_match_the_session_they_came_from():
    """A session short enough not to window must be labelled identically
    either way -- otherwise this change silently shifts every Aalto row, and
    transcription knob fidelity (r=0.994) is the control the diagnosis rests
    on."""
    from typeshi.events import Event
    from typeshi.labels import compute_labels, window_labels

    events = [Event.key("a", 0, 50), Event.key("X", 100, 150),
              Event.backspace(200, 250), Event.key("b", 300, 350)]
    session = compute_labels(events, "ab")
    win = window_labels(events, session)
    assert win.wpm == pytest.approx(session.wpm)
    assert win.corrected_error_rate == pytest.approx(session.corrected_error_rate)
    assert win.revision_rate == pytest.approx(session.revision_rate)


def test_window_labels_survive_a_continuation_windows_cursor_ops():
    """A continuation window's <CUR:p> indexes the buffer as it stood, not the
    window's own text. Replaying the window from empty raises ReplayError --
    caught by a real export: "cursor 651 outside buffer of length 271"."""
    from typeshi.dataset import build_examples
    from typeshi.events import Event
    from typeshi.labels import compute_labels

    events = [Event.key(c, i * 100, i * 100 + 50)
              for i, c in enumerate("abcdef")]
    events += [Event.cursor(5, 700), Event.key("X", 800, 850)]
    session = compute_labels(events, "abcdeXf")

    examples = build_examples("abcdeXf", events, session, "composition",
                              max_events=6)
    assert len(examples) == 2
