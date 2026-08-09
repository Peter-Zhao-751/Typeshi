import numpy as np
import pytest

from typeshi.events import Event
from typeshi.eval.discriminator import (
    featurize, heuristic_baseline, train_discriminator,
)


def _lognormal_session(rng, n=120, mu=4.8, sigma=0.45):
    events, t = [], 0
    for _ in range(n):
        events.append(Event.key("a", t, t + int(rng.lognormal(4.0, 0.3))))
        t += int(rng.lognormal(mu, sigma))
    return events


def test_featurize_returns_fixed_length_vector():
    rng = np.random.default_rng(0)
    a = featurize(_lognormal_session(rng))
    b = featurize(_lognormal_session(rng, n=300))
    assert a.shape == b.shape
    assert np.isfinite(a).all()


def test_featurize_handles_a_degenerate_session():
    """A one-key session has no intervals; the vector must still be finite."""
    v = featurize([Event.key("a", 0, 50)])
    assert np.isfinite(v).all()
    assert v.shape == featurize(_lognormal_session(np.random.default_rng(0))).shape


def test_heuristic_baseline_produces_the_target_text():
    from typeshi.buffer import replay
    events = heuristic_baseline("hello world", wpm=60, seed=0)
    assert replay(events) == "hello world"


def test_heuristic_baseline_is_deterministic_for_a_seed():
    a = heuristic_baseline("the quick brown fox", wpm=60, seed=7)
    b = heuristic_baseline("the quick brown fox", wpm=60, seed=7)
    assert [e.press_time for e in a] == [e.press_time for e in b]


def test_heuristic_baseline_respects_the_requested_speed():
    slow = heuristic_baseline("a" * 200, wpm=30, seed=0)
    fast = heuristic_baseline("a" * 200, wpm=120, seed=0)
    assert slow[-1].press_time > 3 * fast[-1].press_time


def test_discriminator_easily_catches_the_heuristic_baseline():
    """Validates the discriminator has teeth before we trust its verdict."""
    rng = np.random.default_rng(0)
    real = [_lognormal_session(rng) for _ in range(60)]
    fake = [heuristic_baseline("the quick brown fox jumps", wpm=60, seed=i)
            for i in range(60)]
    _, acc = train_discriminator(real, fake, seed=0)
    assert acc > 0.9


def test_discriminator_cannot_separate_identical_distributions():
    """Sanity check: same generator on both sides -> chance accuracy."""
    rng = np.random.default_rng(0)
    a = [_lognormal_session(rng) for _ in range(60)]
    b = [_lognormal_session(rng) for _ in range(60)]
    _, acc = train_discriminator(a, b, seed=0)
    assert acc < 0.65


def test_discriminator_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    real = [_lognormal_session(rng) for _ in range(40)]
    fake = [heuristic_baseline("hello there friend", wpm=55, seed=i)
            for i in range(40)]
    _, first = train_discriminator(real, fake, seed=0)
    _, second = train_discriminator(real, fake, seed=0)
    assert first == second


def test_real_vs_real_control_is_near_chance():
    """A shuffled real-vs-real split must land near chance. A sequential split
    would group whole participants on one side and the classifier would then
    identify typists instead of real-vs-fake."""
    from typeshi.eval.discriminator import real_vs_real_control

    rng = np.random.default_rng(0)
    sessions = [_lognormal_session(rng) for _ in range(60)]
    assert real_vs_real_control(sessions, seed=0) < 0.65


def test_sequential_split_inflates_accuracy_which_is_why_the_control_shuffles():
    """The reason real_vs_real_control shuffles.

    A sequential split of a session list groups whole typists on one side, so
    the classifier separates *people* rather than real-vs-fake. Shuffling the
    same sessions collapses that back to chance. Observed on real data too:
    0.86 sequential vs 0.42 interleaved over 200 Aalto sessions from 14
    participants.
    """
    from typeshi.eval.discriminator import real_vs_real_control

    rng = np.random.default_rng(0)
    fast = [_lognormal_session(rng, mu=4.0) for _ in range(30)]
    slow = [_lognormal_session(rng, mu=6.0) for _ in range(30)]

    _, sequential = train_discriminator(fast, slow, seed=0)
    shuffled = real_vs_real_control(fast + slow, seed=0)

    assert sequential > 0.8, "distinct typist populations are separable"
    assert shuffled < 0.65, "shuffling the same sessions removes the confound"


def test_report_nan_is_serialised_as_null():
    """The heuristic baseline never pauses, so its pause metrics are NaN and
    json.dumps would otherwise emit bare NaN, which is invalid JSON."""
    import json
    import sys

    sys.path.insert(0, "scripts")
    from run_eval import jsonable

    cleaned = jsonable({"pause": {"kl": float("nan"), "frechet": 1.5}, "n": 3})
    assert cleaned["pause"]["kl"] is None
    assert json.loads(json.dumps(cleaned))["pause"]["frechet"] == 1.5
