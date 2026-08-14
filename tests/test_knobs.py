"""Knob fidelity must reward a model that realizes the requested conditions,
expose one that ignores or inverts them, and never crash a report on the
degenerate sweeps (constant knobs) that are its everyday input."""

import math

import pytest

from typeshi.events import Event
from typeshi.eval.knobs import KNOBS, knob_fidelity, realized_labels, sweep_grid
from typeshi.labels import SessionLabels, compute_labels


def _labels(wpm=60.0, ecor=0.0, eunc=0.0, rev=0.0):
    return SessionLabels(wpm, ecor, eunc, rev)


def _type(text, start=0, gap=100, hold=50):
    return [Event.key(c, start + i * gap, start + i * gap + hold)
            for i, c in enumerate(text)]


def test_realized_labels_uses_the_training_label_definition():
    events = _type("a" * 25, gap=100)
    labels = realized_labels(events, "a" * 25)
    assert labels == compute_labels(events, "a" * 25)
    # 25 chars at 100 ms/char -> 120 wpm under the standard formula.
    assert labels.wpm == pytest.approx(120, rel=0.05)
    assert labels.corrected_error_rate == 0.0


def test_perfect_fidelity_scores_r_one_and_mae_zero():
    requested = [_labels(wpm=w, ecor=e)
                 for w in (40, 60, 80, 100) for e in (0.0, 0.05, 0.10)]
    fid = knob_fidelity(requested, list(requested))
    assert fid["wpm"]["r"] == pytest.approx(1.0)
    assert fid["wpm"]["mae"] == pytest.approx(0.0, abs=1e-12)
    assert fid["corrected_error_rate"]["r"] == pytest.approx(1.0)
    assert fid["corrected_error_rate"]["mae"] == pytest.approx(0.0, abs=1e-12)


def test_anticorrelated_realizations_score_r_minus_one():
    requested = [_labels(wpm=w) for w in (40, 60, 80, 100)]
    realized = [_labels(wpm=w) for w in (100, 80, 60, 40)]
    assert knob_fidelity(requested, realized)["wpm"]["r"] == pytest.approx(-1.0)


def test_constant_offset_is_a_calibration_error_not_a_correlation_one():
    requested = [_labels(wpm=w) for w in (40, 60, 80, 100)]
    realized = [_labels(wpm=w + 10) for w in (40, 60, 80, 100)]
    fid = knob_fidelity(requested, realized)
    assert fid["wpm"]["r"] == pytest.approx(1.0)
    assert fid["wpm"]["mae"] == pytest.approx(10.0)


def test_degenerate_variance_gives_nan_r_and_finite_mae():
    # All-same requested WPM: correlation is undefined, calibration is not.
    requested = [_labels(wpm=60.0)] * 5
    realized = [_labels(wpm=w) for w in (50, 55, 60, 65, 70)]
    fid = knob_fidelity(requested, realized)
    assert math.isnan(fid["wpm"]["r"])
    assert fid["wpm"]["mae"] == pytest.approx(6.0)
    # Constant on the realized side is just as undefined.
    fid = knob_fidelity(realized, requested)
    assert math.isnan(fid["wpm"]["r"])


def test_report_covers_every_knob_with_exactly_r_and_mae():
    fid = knob_fidelity([_labels()], [_labels()])
    assert set(fid) == set(KNOBS) == {
        "wpm", "corrected_error_rate",
        "uncorrected_error_rate", "revision_rate",
    }
    for scores in fid.values():
        assert set(scores) == {"r", "mae"}
        assert all(isinstance(v, float) for v in scores.values())


def test_zero_pairs_yield_nan_not_a_crash():
    fid = knob_fidelity([], [])
    assert math.isnan(fid["wpm"]["r"])
    assert math.isnan(fid["wpm"]["mae"])


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        knob_fidelity([_labels()], [])


def test_sweep_grid_covers_the_full_product():
    grid = sweep_grid(_labels(), [40, 80, 120], [0.0, 0.05])
    assert len(grid) == 6
    assert {(g.wpm, g.corrected_error_rate) for g in grid} == {
        (w, e) for w in (40.0, 80.0, 120.0) for e in (0.0, 0.05)
    }


def test_sweep_grid_inherits_unswept_knobs_from_base():
    base = _labels(eunc=0.02, rev=0.1)
    assert sweep_grid(base, [40], [0.0]) == [SessionLabels(40.0, 0.0, 0.02, 0.1)]
    for g in sweep_grid(base, [40, 80], [0.0, 0.1]):
        assert g.uncorrected_error_rate == 0.02
        assert g.revision_rate == 0.1
