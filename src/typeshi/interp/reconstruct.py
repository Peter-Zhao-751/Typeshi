"""Latency matrix -> 2D layout -> score against physical ground truth.

Deliberately model-free: everything here operates on a 27x27 matrix of
press-to-press latencies in milliseconds, whether it came from the model
(digraph.py) or from the corpus (empirical.py). That is what lets the whole
pipeline be validated on a synthetic matrix before any checkpoint is loaded.
"""

from __future__ import annotations

import numpy as np

from typeshi.interp import layout


def to_log_symmetric(latency_ms: np.ndarray) -> np.ndarray:
    """Ordered-pair latencies in ms -> symmetric log-ms, diagonal NaN.

    Log because the bins themselves are geomspaced and typing spans three
    orders of magnitude. The diagonal is the `repeat` class -- a different
    motor event -- so it is masked rather than dragging row and column effects
    toward double-tap speed.
    """
    m = np.log(np.asarray(latency_ms, dtype=float))
    sym = (m + m.T) / 2.0
    np.fill_diagonal(sym, np.nan)
    return sym


def _masked_mean(a: np.ndarray, axis: int) -> np.ndarray:
    """Mean over non-NaN entries, without numpy's empty-slice warning.

    An all-masked row carries no information about that key, so NaN is the
    correct answer rather than a RuntimeWarning -- and test output has to stay
    pristine.
    """
    total = np.nansum(a, axis=axis)
    count = np.sum(~np.isnan(a), axis=axis)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def double_center(m: np.ndarray, iters: int = 50):
    """Iterated two-way mean centering -> (residual, row_effect, col_effect, grand).

    Load-bearing. Some keys are simply slow, and without stripping row and
    column effects the leading component of the residual is "fast keys vs slow
    keys" and geometry never surfaces. NaN-aware, so the masked diagonal and
    any masked thin cells cannot poison a mean.

    Mean, not median. This step first specified Tukey median polish for
    robustness against the same-finger outliers; measured against the synthetic
    matrix it recovers rho=0.61 where mean centering recovers rho=0.90, against
    a ceiling of 0.97. At 27 columns the median is too high-variance to preserve
    a distance signal this small, and the outliers it guarded against are
    removed explicitly by remove_indicator() one step later.

    Robustness lives in the masking instead: every stage is NaN-aware, and
    masking 60 of the 351 unordered cells still recovers rho=0.85. An UNMASKED
    gross outlier does degrade the fit (12 cells at 10x drop it to rho=0.57),
    which is why mask_thin_cells() and log-space averaging upstream are what
    keep this estimator's assumptions true.
    """
    r = np.array(m, dtype=float, copy=True)
    row = np.zeros(r.shape[0])
    col = np.zeros(r.shape[1])
    grand = 0.0
    for _ in range(iters):
        rm = _masked_mean(r, axis=1)
        r -= rm[:, None]
        row += rm
        cm = _masked_mean(r, axis=0)
        r -= cm[None, :]
        col += cm
        # nanmean, not mean: a fully masked key contributes NaN to the
        # accumulator, and a plain mean would spread that NaN across every
        # other key's effect on the next subtraction.
        g = np.nanmean(row)
        row -= g
        grand += g
        g = np.nanmean(col)
        col -= g
        grand += g
    return r, row, col, grand


def remove_indicator(residual: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Subtract the mean residual of the masked cells from those cells.

    A one-term regression. The same-finger penalty is close to a constant
    additive cost in log-ms; leaving it in makes MDS push physically ADJACENT
    keys apart, which is the failure mode §2 of the spec predicts.
    """
    out = np.array(residual, dtype=float, copy=True)
    sel = mask & ~np.isnan(out)
    if sel.any():
        out[sel] -= np.nanmean(out[sel])
    return out


def embed_2d(residual: np.ndarray, seed: int = 0) -> np.ndarray:
    """Residual interaction -> 2D coordinates via metric MDS (SMACOF).

    MDS needs a non-negative dissimilarity with a zero diagonal; the residual
    is centered and has negatives, so it is shifted by its own minimum. NaN
    cells are filled at the residual mean -- SMACOF has no missing-data mode,
    and filling at the mean makes a masked cell exert no pull either way.
    """
    from sklearn.manifold import MDS

    d = np.array(residual, dtype=float, copy=True)
    off = ~np.eye(d.shape[0], dtype=bool)
    d[np.isnan(d)] = np.nanmean(d[off])
    d = d - d[off].min()
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    mds = MDS(
        n_components=2,
        metric="precomputed",
        init="random",
        random_state=seed,
        normalized_stress=False,
        n_init=8,
    )
    return mds.fit_transform(d)


def align(coords: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, float]:
    """Procrustes-align `coords` onto `truth`; returns coordinates in key-widths.

    scipy's procrustes standardizes both inputs to unit Frobenius norm, which
    would report position error in arbitrary units. Rescaling by the truth's
    own norm puts the error back into key-widths -- the only unit a reader can
    interpret. Reflection is allowed: a mirrored keyboard is still a recovered
    keyboard, since nothing in the data fixes chirality.
    """
    from scipy.spatial import procrustes

    _, standardized_fit, disparity = procrustes(truth, coords)
    center = truth.mean(axis=0)
    scale = np.linalg.norm(truth - center)
    return standardized_fit * scale + center, float(disparity)


def neighbor_recall(fitted: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of each key's touching neighbours that survive reconstruction.

    A key's true neighbours are those within 1.05u: in-row neighbours are
    exactly 1.0u apart and the stagger puts the nearest cross-row key at
    ~1.03u, so the threshold picks out exactly the ring of touching keys.
    """
    from scipy.spatial.distance import squareform, pdist

    true_d = squareform(pdist(truth))
    fit_d = squareform(pdist(fitted))
    recalls = []
    for i in range(len(truth)):
        true_nb = {j for j in range(len(truth)) if j != i and true_d[i, j] <= 1.05}
        if not true_nb:
            continue
        order = [j for j in np.argsort(fit_d[i]) if j != i][: len(true_nb)]
        recalls.append(len(true_nb & set(order)) / len(true_nb))
    return float(np.mean(recalls))


def hand_accuracy(fitted: np.ndarray, keys=layout.KEYS) -> float:
    """Leave-one-out accuracy of a linear left/right split of the fitted map.

    Leave-one-out rather than a train/test split: 26 points is far too few to
    hold any out, and a resubstitution score on a 2D linear model would be
    optimistic enough to be meaningless.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    fingers = layout.finger_of()
    keep = np.array([fingers[k][0] in ("L", "R") for k in keys])
    y = np.array([fingers[k][0] == "R" for k in keys])[keep]
    return float(
        cross_val_score(
            LogisticRegression(), fitted[keep], y, cv=LeaveOneOut()
        ).mean()
    )


def score(fitted: np.ndarray, truth: np.ndarray, keys=layout.KEYS) -> dict[str, float]:
    """Level-2 and level-3 metrics from the spec's scoring section."""
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    return {
        "distance_spearman": float(spearmanr(pdist(fitted), pdist(truth)).statistic),
        "mean_position_error_u": float(
            np.mean(np.linalg.norm(fitted - truth, axis=1))
        ),
        "neighbor_recall": neighbor_recall(fitted, truth),
        "hand_accuracy": hand_accuracy(fitted, keys),
    }


def permutation_p(coords: np.ndarray, truth: np.ndarray, n: int = 500,
                  seed: int = 0) -> float:
    """P(a random key labelling aligns at least this well).

    Guards against reading a good-looking Procrustes fit as significant when
    27 points in 2D would land that way by chance.
    """
    rng = np.random.default_rng(seed)
    _, observed = align(coords, truth)
    at_least_as_good = 0
    for _ in range(n):
        _, d = align(coords[rng.permutation(len(truth))], truth)
        at_least_as_good += d <= observed
    return float((at_least_as_good + 1) / (n + 1))


def synthetic_latency(seed: int = 0, same_finger_penalty: float = 0.35,
                      noise: float = 0.02) -> np.ndarray:
    """A latency matrix generated FROM the truth, for validating the pipeline.

    log latency = grand + row[a] + col[b] + beta*distance
                  + penalty*same_finger + noise

    Deliberately includes both confounds the real analysis must survive: large
    per-key speed effects, and the same-finger penalty that inverts the
    distance signal for adjacent keys.
    """
    rng = np.random.default_rng(seed)
    truth = layout.truth_coords()
    n = len(layout.KEYS)
    row = rng.normal(0, 0.25, n)
    grand, beta = np.log(180.0), 0.06
    out = np.zeros((n, n))
    for i, a in enumerate(layout.KEYS):
        for j, b in enumerate(layout.KEYS):
            dist = float(np.linalg.norm(truth[i] - truth[j]))
            cls = layout.bigram_class(a, b)
            log_ms = grand + row[i] + row[j] + beta * dist
            if cls == "same_finger":
                log_ms += same_finger_penalty
            if cls == "repeat":
                log_ms -= 0.2
            out[i, j] = np.exp(log_ms + rng.normal(0, noise))
    return out
