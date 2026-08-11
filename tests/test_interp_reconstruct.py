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
