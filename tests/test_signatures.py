"""Pins the Tier-2 composition-signature math on constructed sessions.

Sessions are synthetic with known structure -- boundary pauses at 3x/9x the
within-word gap, bursts of scripted lengths, scripted BKSP runs -- so every
expected value below is hand-countable.
"""

import json
import math

import numpy as np
import pytest

from typeshi.events import Event
from typeshi.eval.signatures import (
    compare_signatures, p_bursts, pause_at_boundaries, pooled_signatures,
    revision_ops,
)

HOLD = 40


def _type(text, gaps):
    """One KEY event per char; gaps[i] is the interval before char i
    (gaps[0] positions the first press and produces no interval)."""
    events, t = [], 0.0
    for ch, g in zip(text, gaps, strict=True):
        t += g
        events.append(Event.key(ch, t, t + HOLD))
    return events


# Gap encodes the class of the key it precedes: 100 within-word, 300 at a
# word boundary, 900 after ". " -- a clause boundary.
TEXT = "ab. cd ef"
GAPS = [0, 100, 100, 100, 900, 100, 100, 300, 100]


def test_pause_classes_split_by_position_in_buffer():
    f = pause_at_boundaries(_type(TEXT, GAPS))
    assert list(f["clause_boundary"]) == [900]
    assert list(f["word_boundary"]) == [300]
    assert list(f["within_word"]) == [100] * 6  # incl. the '.' and spaces


def test_comma_opens_a_clause_boundary():
    f = pause_at_boundaries(_type("x, y", [0, 100, 100, 900]))
    assert list(f["clause_boundary"]) == [900]
    assert f["word_boundary"].size == 0


def test_classification_reads_the_buffer_not_the_typed_order():
    """After 'a', ' ', BKSP the buffer is 'a' again: the next key is
    within-word even though the most recently TYPED char was a space."""
    events = [
        Event.key("a", 0, HOLD),
        Event.key(" ", 100, 100 + HOLD),
        Event.backspace(200, 200 + HOLD),
        Event.key("b", 500, 500 + HOLD),
    ]
    f = pause_at_boundaries(events)
    # The interval into the BKSP itself belongs to no class.
    assert list(f["within_word"]) == [100, 300]
    assert f["word_boundary"].size == 0 and f["clause_boundary"].size == 0


def test_offset_zero_falls_into_the_within_word_catch_all():
    events = [
        Event.key("a", 0, HOLD),
        Event.cursor(0, 200),
        Event.key("b", 300, 300 + HOLD),
    ]
    f = pause_at_boundaries(events)
    assert list(f["within_word"]) == [100]


def test_degenerate_sessions_have_no_intervals():
    for events in ([], [Event.key("a", 0, HOLD)]):
        f = pause_at_boundaries(events)
        assert all(v.size == 0 for v in f.values())


def test_p_burst_lengths_are_runs_between_two_second_pauses():
    gaps = [0, 100, 100, 2500, 100, 100, 100, 2500, 100]
    b = p_bursts(_type("abcdefghi", gaps))
    assert list(b) == [3, 4, 2]
    assert sum(b) == 9  # every keystroke lands in exactly one burst


def test_p_burst_threshold_is_strict_and_parametric():
    events = _type("abc", [0, 2000, 2500])
    assert list(p_bursts(events)) == [2, 1]  # 2000 is not > 2000
    assert list(p_bursts(events, threshold_ms=3000)) == [3]


def test_p_burst_degenerates_stay_finite():
    assert p_bursts([]).size == 0
    assert list(p_bursts([Event.key("a", 0, HOLD)])) == [1]


def test_revision_ops_counts_and_episode_lengths():
    events = [
        Event.key("a", 0, HOLD),
        Event.backspace(100, 100 + HOLD),
        Event.backspace(200, 200 + HOLD),  # run of 2
        Event.key("b", 300, 300 + HOLD),
        Event.cursor(0, 400),
        Event.key("c", 500, 500 + HOLD),
        Event.seldel(0, 1, 600),
        Event.backspace(700, 700 + HOLD),  # trailing run of 1 still flushed
    ]
    ops = revision_ops(events)
    assert ops["bksp_frac"] == pytest.approx(3 / 8)
    assert ops["cursor_count"] == 1
    assert ops["seldel_count"] == 1
    assert list(ops["revision_episode_lengths"]) == [2, 1]


def test_revision_ops_empty_session_is_all_zero():
    ops = revision_ops([])
    assert ops["bksp_frac"] == 0.0
    assert ops["cursor_count"] == 0 and ops["seldel_count"] == 0
    assert ops["revision_episode_lengths"].size == 0


def test_pooling_is_per_session_never_concatenation():
    """Every session is rebased to zero, so concatenating event lists
    fabricates a negative interval at each boundary; the pooled path
    computes per session first (distributional.pooled_features rationale)."""
    sessions = [_type(TEXT, GAPS) for _ in range(3)]
    pooled = pooled_signatures(sessions)
    assert pooled["pause_clause_boundary"].size == 3
    assert pooled["pause_within_word"].size == 18
    assert all((v >= 0).all() for v in pooled.values())
    assert pooled["bksp_frac"].size == 3  # one scalar per session

    flat = pause_at_boundaries([e for s in sessions for e in s])
    assert (flat["within_word"] < 0).any(), "naive concatenation should break"


def test_pooled_signatures_skips_empty_sessions():
    pooled = pooled_signatures([[], _type("ab", [0, 100]), []])
    assert pooled["bksp_frac"].size == 1
    assert pooled["p_burst"].size == 1


# --- compare_signatures on synthetic collections ---------------------------

_C_TEXT = "ab. cd ef, gh ij. "
_C_GAPS = [0, 100, 100, 100, 900, 100, 100, 300, 100,
           100, 100, 900, 100, 100, 300, 100, 100, 100]


def _composition_session(seed, scale=1.0):
    """Deterministic composition-shaped session: two clause boundaries, two
    word boundaries, one BKSP episode of 2, one cursor move. Gaps are
    jittered by the seeded rng and stretched by `scale`."""
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.8, 1.25, size=len(_C_GAPS))
    events = _type(_C_TEXT, [g * j * scale for g, j in zip(_C_GAPS, jitter)])
    t = events[-1].press_time
    return events + [
        Event.backspace(t + 150 * scale, t + 150 * scale + HOLD),
        Event.backspace(t + 300 * scale, t + 300 * scale + HOLD),
        Event.cursor(16, t + 500 * scale),
        Event.key("x", t + 700 * scale, t + 700 * scale + HOLD),
        Event.key("y", t + 800 * scale, t + 800 * scale + HOLD),
    ]


ALL_CLASSES = {
    "pause_clause_boundary", "pause_word_boundary", "pause_within_word",
    "p_burst", "revision_episode_lengths",
    "bksp_frac", "cursor_count", "seldel_count",
}


def test_compare_signatures_scores_every_class_finite_or_nan():
    real = [_composition_session(seed) for seed in range(12)]
    fake = [_composition_session(seed + 100) for seed in range(12)]
    result = compare_signatures(real, fake)
    assert set(result) == ALL_CLASSES
    for scores in result.values():
        assert set(scores) == {"kl", "frechet"}
        for v in scores.values():
            assert isinstance(v, float) and not math.isinf(v)


def test_compare_signatures_near_zero_for_identical_collections():
    real = [_composition_session(seed) for seed in range(12)]
    fake = [_composition_session(seed) for seed in range(12)]
    result = compare_signatures(real, fake)
    for key in ("pause_clause_boundary", "pause_word_boundary",
                "pause_within_word", "p_burst", "revision_episode_lengths"):
        assert result[key]["kl"] == pytest.approx(0.0, abs=1e-6), key
        assert result[key]["frechet"] == pytest.approx(0.0, abs=1e-9), key


def test_compare_signatures_detects_stretched_boundaries():
    real = [_composition_session(seed) for seed in range(12)]
    matched = [_composition_session(seed + 100) for seed in range(12)]
    slowed = [_composition_session(seed + 100, scale=3.0) for seed in range(12)]
    near = compare_signatures(real, matched)
    far = compare_signatures(real, slowed)
    for key in ("pause_clause_boundary", "pause_word_boundary"):
        assert far[key]["kl"] > near[key]["kl"]
        assert far[key]["frechet"] > near[key]["frechet"]


def test_compare_signatures_degenerate_collections_stay_nan_safe():
    """One-event sessions have no intervals and too few scalars; every
    metric refuses with NaN instead of crashing or feigning precision."""
    ones = [[Event.key("a", 0, HOLD)] for _ in range(3)]
    for real, fake in ((ones, ones), ([[]], [[]])):
        result = compare_signatures(real, fake)
        assert set(result) == ALL_CLASSES
        assert all(math.isnan(v) for s in result.values() for v in s.values())


def test_compare_signatures_is_jsonable():
    real = [_composition_session(seed) for seed in range(12)]
    result = compare_signatures(real, real)
    json.dumps(result)  # np.float64 anywhere in the dict would raise
