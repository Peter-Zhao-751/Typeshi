import numpy as np
import pytest

from typeshi.events import Event
from typeshi.eval.distributional import (
    compare, frechet_distance, kl_divergence, timing_features,
)


def _session(gaps, hold=50):
    """Builds one key per gap. Note the final gap dangles: N gaps produce N
    events and therefore only N-1 inter-key intervals."""
    events, t = [], 0
    for g in gaps:
        events.append(Event.key("a", t, t + hold))
        t += g
    return events


def test_timing_features_extracts_iki_and_hold():
    f = timing_features(_session([100, 120, 140]))
    assert list(f["iki"]) == [100, 120]
    assert list(f["hold"]) == [50, 50, 50]


def test_pauses_are_gaps_over_one_second():
    # Trailing 100 so the 3000 ms gap becomes a realised interval.
    f = timing_features(_session([100, 5000, 100, 3000, 100]))
    assert sorted(f["pause"]) == [3000, 5000]


def test_bursts_are_key_runs_between_pauses():
    f = timing_features(_session([100, 100, 5000, 100, 100, 100]))
    # Six keys, one pause after the third: two runs of three.
    assert sorted(f["burst"]) == [3, 3]
    assert sum(f["burst"]) == 6


def test_burst_lengths_always_account_for_every_key():
    for gaps in ([100] * 9, [5000] * 9, [100, 5000, 100, 5000, 100]):
        f = timing_features(_session(gaps))
        assert sum(f["burst"]) == len(gaps)


def test_kl_of_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    x = rng.lognormal(5, 0.4, 4000)
    y = rng.lognormal(5, 0.4, 4000)
    assert kl_divergence(x, y) < 0.05


def test_kl_grows_when_distributions_differ():
    rng = np.random.default_rng(0)
    x = rng.lognormal(5, 0.4, 4000)
    y = rng.lognormal(6, 0.4, 4000)
    assert kl_divergence(x, y) > kl_divergence(x, x + 1)


def test_kl_is_symmetric():
    rng = np.random.default_rng(0)
    x = rng.lognormal(5, 0.4, 2000)
    y = rng.lognormal(5.5, 0.5, 2000)
    assert kl_divergence(x, y) == pytest.approx(kl_divergence(y, x), rel=1e-9)


def test_frechet_is_zero_for_identical_samples():
    x = np.linspace(1.0, 40.0, 12)
    assert frechet_distance(x, x) == pytest.approx(0.0, abs=1e-9)


def test_frechet_grows_with_separation():
    x = np.linspace(100.0, 130.0, 12)
    near = frechet_distance(x, x + 5)
    far = frechet_distance(x, x + 500)
    assert far > near


def test_frechet_refuses_singletons_instead_of_reporting_perfect_zero():
    """A singleton against itself used to score a finite 0.0 -- a misleading
    'perfect' result from one observation. Now NaN, like KL."""
    assert np.isnan(frechet_distance([5.0], [5.0]))
    assert np.isnan(frechet_distance(list(range(5)), list(range(5))))


def test_metrics_reject_impossible_but_tolerate_zero_timings():
    """Zero holds occur in real Aalto data (release clamped to press, ~1 in
    38k events); rejecting them crashed the whole eval. Negative and NaN
    stay hard errors in BOTH metrics."""
    from typeshi.eval.distributional import kl_divergence

    ok = [0.0] + [float(v) for v in range(1, 15)]
    assert np.isfinite(kl_divergence(ok, ok))
    assert np.isfinite(frechet_distance(ok, ok))
    for bad in ([-1.0] * 15, [float("nan")] * 15, [float("inf")] * 15):
        with pytest.raises(ValueError):
            kl_divergence(bad, ok)
        with pytest.raises(ValueError):
            frechet_distance(bad, ok)


def test_all_empty_collections_score_nan_not_crash():
    from typeshi.eval.distributional import compare_sessions

    result = compare_sessions([[], []], [[], []])
    assert all(np.isnan(v["kl"]) and np.isnan(v["frechet"])
               for v in result.values())


def test_empty_input_yields_nan_not_a_crash():
    assert np.isnan(kl_divergence([], [1.0, 2.0]))
    assert np.isnan(frechet_distance([1.0], []))


def test_compare_reports_every_feature():
    real, fake = _session([100] * 50), _session([110] * 50)
    result = compare(real, fake)
    assert set(result) == {"iki", "hold", "pause", "burst"}
    assert "kl" in result["iki"] and "frechet" in result["iki"]


def test_pooling_sessions_does_not_invent_negative_intervals():
    """Every session is rebased to zero, so concatenating events before
    measuring fabricates a large negative gap at each session boundary."""
    from typeshi.eval.distributional import pooled_features

    sessions = [_session([100, 120, 140]) for _ in range(3)]
    pooled = pooled_features(sessions)
    assert (pooled["iki"] > 0).all()
    assert pooled["iki"].size == 6  # 2 intervals per session, not 8

    naive = timing_features([e for s in sessions for e in s])
    assert (naive["iki"] < 0).any(), "naive concatenation should be broken"


@pytest.mark.filterwarnings("ignore:invalid value encountered in log1p")
def test_flat_concatenation_is_rejected_loudly_where_pooling_works():
    """Concatenating rebased sessions fabricates negative IKIs. That used to
    surface as a silent NaN; kl_divergence now rejects it outright, while the
    pooled path stays finite."""
    from typeshi.eval.distributional import compare_sessions, kl_divergence

    sessions = [_session([100, 120, 140, 160]) for _ in range(4)]
    flat = [e for s in sessions for e in s]
    flat_iki = timing_features(flat)["iki"]
    with pytest.raises(ValueError):
        kl_divergence(flat_iki, flat_iki)
    pooled = compare_sessions(sessions, sessions)
    assert pooled["iki"]["frechet"] == pytest.approx(0.0, abs=1e-9)


def test_kl_refuses_tiny_samples_where_smoothing_dominates():
    from typeshi.eval.distributional import kl_divergence

    assert np.isnan(kl_divergence([5.0], [7.0]))
    assert np.isnan(kl_divergence(list(range(1, 10)), list(range(1, 50))))


def test_empty_session_has_no_burst():
    f = timing_features([])
    assert f["burst"].size == 0


def test_pooled_features_ignores_empty_sessions():
    from typeshi.eval.distributional import pooled_features

    pooled = pooled_features([[], _session([100, 100]), []])
    assert pooled["iki"].size == 1
