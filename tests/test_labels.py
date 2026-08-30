import pytest
from typeshi.events import Event
from typeshi.labels import SessionLabels, compute_labels


def _type(text, start=0, gap=100, hold=50):
    return [Event.key(c, start + i * gap, start + i * gap + hold)
            for i, c in enumerate(text)]


def test_wpm_matches_standard_formula():
    # 25 chars at exactly 100 ms/char = 2.5 s -> (25/5) / (2.5/60) = 120 wpm
    events = _type("a" * 25, gap=100)
    labels = compute_labels(events, "a" * 25)
    assert labels.wpm == pytest.approx(120, rel=0.05)


def test_clean_session_has_zero_error_rates():
    events = _type("hello")
    labels = compute_labels(events, "hello")
    assert labels.corrected_error_rate == 0.0
    assert labels.uncorrected_error_rate == 0.0


def test_typo_then_backspace_counts_as_corrected_error():
    events = _type("helz") + [Event.backspace(400, 450)] + _type("lo", start=500)
    labels = compute_labels(events, "hello")
    assert labels.corrected_error_rate > 0
    assert labels.uncorrected_error_rate == 0.0


def test_uncorrected_typo_shows_in_uncorrected_rate():
    events = _type("helzo")
    labels = compute_labels(events, "hello")
    assert labels.uncorrected_error_rate > 0
    assert labels.corrected_error_rate == 0.0


def test_revision_rate_counts_cursor_and_seldel():
    events = _type("hello") + [Event.cursor(0, 600), Event.seldel(0, 1, 700)]
    labels = compute_labels(events, "ello")
    assert labels.revision_rate == pytest.approx(2 / 7, rel=0.01)


def test_knob_tokens_are_stable_and_cover_all_knobs():
    from typeshi.serialize import rev_bin

    labels = SessionLabels(wpm=61.4, corrected_error_rate=0.031,
                           uncorrected_error_rate=0.004, revision_rate=0.12)
    # REV is on its own geometric scale -- whole percents put the median
    # composition window in bin 0 next to windows that never revise.
    assert labels.to_tokens("transcription") == (
        f"<MODE:T><WPM:12><ECOR:3><EUNC:0><REV:{rev_bin(0.12)}>"
    )
    assert rev_bin(0.12) != 12, "the scale must not coincide with percents"
    assert labels.to_tokens("composition").startswith("<MODE:C>")


def test_extreme_knob_values_clamp_instead_of_overflowing():
    labels = SessionLabels(wpm=9999, corrected_error_rate=1.0,
                           uncorrected_error_rate=1.0, revision_rate=1.0)
    assert labels.to_tokens("transcription") == (
        "<MODE:T><WPM:39><ECOR:30><EUNC:30><REV:30>"
    )


def test_empty_session_does_not_divide_by_zero():
    labels = compute_labels([], "")
    assert labels.wpm == 0.0
