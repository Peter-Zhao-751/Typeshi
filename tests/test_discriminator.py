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


def test_eval_refuses_to_run_without_the_writer_split(tmp_path):
    """Tier-1 is meaningless if scored on writers the model trained on, so a
    missing split file must be a hard stop rather than a silent full sweep."""
    import sys

    sys.path.insert(0, "scripts")
    from run_eval import load_test_writers

    missing = tmp_path / "split.json"
    with pytest.raises(SystemExit):
        load_test_writers(missing, allow_unsplit=False)
    assert load_test_writers(missing, allow_unsplit=True) is None


def test_eval_reads_the_held_out_writers(tmp_path):
    import json
    import sys

    sys.path.insert(0, "scripts")
    from run_eval import load_test_writers

    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "train_writers": ["aalto:1", "aalto:2"], "test_writers": ["aalto:3"],
    }))
    assert load_test_writers(path, allow_unsplit=False) == {"aalto:3"}


def test_paired_cv_scores_exact_copies_at_chance_not_below():
    """THE showstopper: with unpaired folds, an exact copy of each real
    session in the fake class scored 0.085 -- and a gate of 'accuracy <=
    0.55' rewards that. Grouped folds keep each pair together, restoring the
    definitionally correct 0.5."""
    rng = np.random.default_rng(0)
    real = [_lognormal_session(rng) for _ in range(100)]
    copies = [list(s) for s in real]

    _, unpaired = train_discriminator(real, copies, seed=0)
    _, paired = train_discriminator(real, copies, seed=0, paired=True)
    assert unpaired < 0.30, "the leak this guards against has disappeared?"
    assert 0.40 <= paired <= 0.60


def test_paired_cv_requires_equal_counts():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        train_discriminator(
            [_lognormal_session(rng)] * 4, [_lognormal_session(rng)] * 3,
            paired=True,
        )


def test_shuffled_timing_preserves_gap_multiset_and_chars():
    from typeshi.eval.discriminator import shuffle_timing

    rng = np.random.default_rng(0)
    s = _lognormal_session(rng, n=50)
    sh = shuffle_timing(s, seed=1)
    gaps = sorted(b.press_time - a.press_time for a, b in zip(s, s[1:]))
    sgaps = sorted(b.press_time - a.press_time for a, b in zip(sh, sh[1:]))
    assert gaps == sgaps
    assert [e.char for e in sh] == [e.char for e in s]
    holds = [e.release_time - e.press_time for e in s]
    sholds = [e.release_time - e.press_time for e in sh]
    assert holds == sholds


def test_timing_only_features_are_shorter_than_full():
    rng = np.random.default_rng(0)
    s = _lognormal_session(rng)
    assert featurize(s, count_features=False).size == featurize(s).size - 4


def test_eval_gate_rejects_wrong_text_generations(tmp_path):
    """Valid grammar over the wrong characters must not count as a valid
    generation -- realistic timings for 'aaaa' are not a transcription of
    the target."""
    import sys

    sys.path.insert(0, "scripts")
    from run_eval import transcription_generation_ok

    from typeshi.events import Event

    target = "hello world"
    good = [Event.key(c, i * 100, i * 100 + 50) for i, c in enumerate(target)]
    garbage = [Event.key("a", i * 100, i * 100 + 50) for i in range(len(target))]
    with_cursor = good[:-1] + [Event.cursor(0, 99999)]
    assert transcription_generation_ok(good, target)
    assert not transcription_generation_ok(garbage, target)
    assert not transcription_generation_ok(with_cursor, target)
    assert not transcription_generation_ok([], target)


def test_split_loader_rejects_overlapping_writer_sets(tmp_path):
    import json
    import sys

    sys.path.insert(0, "scripts")
    from run_eval import load_test_writers

    path = tmp_path / "split.json"
    path.write_text(json.dumps({
        "train_writers": ["aalto:1", "aalto:2"],
        "test_writers": ["aalto:2", "aalto:3"],
    }))
    with pytest.raises(SystemExit):
        load_test_writers(path, allow_unsplit=False)
