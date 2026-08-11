import warnings

import numpy as np
import pytest
from typeshi.interp import layout, reconstruct


def test_log_symmetric_masks_the_diagonal():
    # The diagonal is the `repeat` class -- a double tap is a different motor
    # event from a two-key move, so it must not drag the row/column effects.
    m = np.full((27, 27), 200.0)
    out = reconstruct.to_log_symmetric(m)
    assert np.isnan(np.diag(out)).all()
    assert out[0, 1] == pytest.approx(np.log(200.0))


def test_double_center_strips_planted_row_and_column_effects():
    rng = np.random.default_rng(0)
    interaction = rng.normal(0, 0.05, (27, 27))
    interaction = (interaction + interaction.T) / 2
    row = rng.normal(0, 0.5, 27)
    planted = interaction + row[:, None] + row[None, :]
    np.fill_diagonal(planted, np.nan)
    residual, row_eff, _, _ = reconstruct.double_center(planted)
    # Row effects should be recovered up to a shared constant, so their
    # SPREAD is what must match -- an absolute comparison would fail on the
    # grand-effect split alone.
    assert np.std(row_eff - row) < 0.1
    assert np.nanstd(residual) < np.nanstd(planted)


def test_a_fully_masked_key_does_not_poison_the_other_effects():
    """A key whose every pairing is masked has no measurable speed of its own.

    Its own effect must come back NaN while the other 26 survive. A plain mean
    in the grand-splitting step spreads that one NaN across all of them, and
    warns while doing it -- both of which this test refuses.
    """
    rng = np.random.default_rng(0)
    interaction = rng.normal(0, 0.05, (27, 27))
    interaction = (interaction + interaction.T) / 2
    row = rng.normal(0, 0.5, 27)
    planted = interaction + row[:, None] + row[None, :]
    np.fill_diagonal(planted, np.nan)
    planted[5, :] = np.nan
    planted[:, 5] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # an empty-slice warning fails the test
        _, row_eff, _, _ = reconstruct.double_center(planted)
    assert np.isnan(row_eff[5])
    assert np.isfinite(np.delete(row_eff, 5)).all()


def test_pipeline_recovers_a_keyboard_from_a_synthetic_matrix():
    """The test that makes a null result interpretable.

    If a matrix GENERATED from the true coordinates does not reconstruct, the
    analysis is broken and no statement about the model is possible. Do not
    lower these thresholds to make it pass -- debug the pipeline.
    """
    truth = layout.truth_coords()
    latency = reconstruct.synthetic_latency(seed=0)
    sym = reconstruct.to_log_symmetric(latency)
    residual, _, _, _ = reconstruct.double_center(sym)
    # Ground-truth same-finger removal: this test validates the geometry path,
    # not the blind detector (that is Task 3).
    penalty = np.zeros_like(residual, dtype=bool)
    for i, a in enumerate(layout.KEYS):
        for j, b in enumerate(layout.KEYS):
            penalty[i, j] = layout.bigram_class(a, b) == "same_finger"
    residual = reconstruct.remove_indicator(residual, penalty)
    fitted, _ = reconstruct.align(reconstruct.embed_2d(residual, seed=0), truth)
    metrics = reconstruct.score(fitted, truth)
    assert metrics["distance_spearman"] > 0.85
    assert metrics["mean_position_error_u"] < 1.5


def test_masked_cells_do_not_break_the_reconstruction():
    """Robustness lives in the masking, not in the estimator.

    double_center() uses the mean, which is precise but not outlier-resistant.
    The contract that makes that safe is that thin and unreliable cells are
    masked to NaN upstream and every stage here is NaN-aware. This pins that
    contract: 30 masked cells of 351 must still reconstruct.
    """
    truth = layout.truth_coords()
    latency = reconstruct.synthetic_latency(seed=0)
    symmetric = reconstruct.to_log_symmetric(latency)
    rng = np.random.default_rng(100)
    for i, j in rng.choice(len(layout.KEYS), size=(30, 2)):
        if i != j:
            symmetric[i, j] = np.nan
            symmetric[j, i] = np.nan
    residual, _, _, _ = reconstruct.double_center(symmetric)
    penalty = np.zeros_like(residual, dtype=bool)
    for i, a in enumerate(layout.KEYS):
        for j, b in enumerate(layout.KEYS):
            penalty[i, j] = layout.bigram_class(a, b) == "same_finger"
    residual = reconstruct.remove_indicator(residual, penalty)
    fitted, _ = reconstruct.align(reconstruct.embed_2d(residual, seed=0), truth)
    assert reconstruct.score(fitted, truth)["distance_spearman"] > 0.85


def test_permutation_test_rejects_a_random_labelling():
    truth = layout.truth_coords()
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 3.0, truth.shape)
    assert reconstruct.permutation_p(noise, truth, n=200, seed=0) > 0.05


def test_blind_detector_finds_planted_same_finger_pairs():
    # No ground truth reaches the detector -- it sees only the residual.
    latency = reconstruct.synthetic_latency(seed=0)
    sym = reconstruct.to_log_symmetric(latency)
    residual, _, _, _ = reconstruct.double_center(sym)
    truth_mask = reconstruct.true_same_finger()
    # The AUC is the strong claim: the residual RANKS same-finger pairs to the
    # top. Where the detector cuts that ranking is a separate, weaker question.
    assert reconstruct.same_finger_auc(residual, truth_mask) > 0.9
    detected = reconstruct.detect_same_finger(residual)
    assert detected.shape == (27, 27)
    assert (detected == detected.T).all()   # symmetric, like the residual
    # Calibration without being told the count: 41 of the 351 unordered pairs
    # are truly same-finger, and the rule must land in that neighbourhood
    # rather than flagging the whole upper half of the distribution.
    upper = np.triu_indices(27, k=1)
    assert 25 <= detected[upper].sum() <= 70


def test_both_modes_reconstruct_the_synthetic_keyboard():
    latency = reconstruct.synthetic_latency(seed=0)
    blind = reconstruct.reconstruct(latency, mode="blind", seed=0)
    aware = reconstruct.reconstruct(latency, mode="finger_aware", seed=0)
    # Blind measures 0.797-0.851 across seeds against finger-aware's
    # 0.893-0.904 -- the price of using no ground truth, recorded rather than
    # wished away. The bar sits below the observed minimum with margin.
    assert blind["metrics"]["distance_spearman"] > 0.75
    assert aware["metrics"]["distance_spearman"] > 0.85
    # Ground truth can only help; if blind beats aware something is wrong.
    assert aware["metrics"]["distance_spearman"] > blind["metrics"]["distance_spearman"]


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        reconstruct.reconstruct(reconstruct.synthetic_latency(), mode="magic")
