# Keyboard Reconstruction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the physical layout of a QWERTY keyboard from `motor-tiny`, a 19M model that has never seen a key coordinate — only event streams like `<a:54><DT:46><s:54>`.

**Architecture:** Four independent modules under a new `src/typeshi/interp/` package. `layout.py` holds physical ground truth (pure data). `reconstruct.py` turns any 27×27 latency matrix into a 2D layout and scores it. `empirical.py` measures that matrix from the corpus. `digraph.py` reads it off the model's `<DT:k>` softmax by teacher-forcing a fixed carrier. `scripts/keyboard_probe.py` wires them together with controls. Nothing imports from or changes the training or eval paths.

**Tech Stack:** numpy, scipy (`procrustes`, `spearmanr`, `pdist`), scikit-learn (`MDS`, `GaussianMixture`, `LogisticRegression`, `roc_auc_score`), transformers + torch for the model probe, matplotlib for figures (new optional extra).

**Spec:** `docs/superpowers/specs/2026-08-11-keyboard-reconstruction-design.md`

## Global Constraints

- Python ≥ 3.11. `numpy`, `scipy`, `scikit-learn` are already base dependencies; matplotlib is added as a new `viz` optional extra and must only be imported inside the figure code path.
- **CPU by default.** A training run owns the MPS device; a probe that steals it slows the real work. Same precedent as `scripts/playground.py`.
- Tests run offline with no network and no real corpora (`tests/conftest.py`). Corpus-touching tests use fixtures under `tests/fixtures/`.
- Import constants from `typeshi.config` — never inline `128`, `1`, or `120_000`.
- `layout.KEYS` — 26 lowercase letters plus space, 27 entries — indexes every matrix in this package, in that order.
- Do not change training, eval, serialization, or grammar behaviour. Exactly one edit to existing code is permitted: adding a public `encode_char()` to `src/typeshi/serialize.py` (Task 5).
- Test style follows the repo: plain `pytest` functions, descriptive names, comments that explain *why* a check exists.
- Commit at the end of every task.

---

### Task 1: Physical ground truth

Every metric in this package is scored against this file, so a wrong entry silently invalidates the entire result. The tests hand-check individual keys rather than round-tripping the tables against themselves.

**Files:**
- Create: `src/typeshi/interp/__init__.py`
- Create: `src/typeshi/interp/layout.py`
- Test: `tests/test_interp_layout.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `KEYS: tuple[str, ...]` (27), `key_positions() -> dict[str, tuple[float, float]]`, `truth_coords() -> np.ndarray` shape `(27, 2)` in `KEYS` order, `finger_of() -> dict[str, tuple[str, int]]`, `bigram_class(a: str, b: str) -> str` returning one of `"repeat" | "same_finger" | "same_hand" | "alternate"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interp_layout.py
import numpy as np
import pytest
from typeshi.interp import layout


def test_twenty_seven_keys_all_placed_and_assigned():
    assert len(layout.KEYS) == 27
    assert layout.KEYS[-1] == " "
    pos, fingers = layout.key_positions(), layout.finger_of()
    assert all(k in pos and k in fingers for k in layout.KEYS)


@pytest.mark.parametrize("key,expected", [
    ("q", (2.00, 1.0)),    # Tab is 1.5u wide, so Q's center lands at 2.0
    ("p", (11.00, 1.0)),   # ninth key along the top row
    ("a", (2.25, 0.0)),    # Caps is 1.75u
    ("l", (10.25, 0.0)),
    ("z", (2.75, -1.0)),   # LShift is 2.25u
    ("m", (8.75, -1.0)),
    (" ", (6.875, -2.0)),  # 6.25u space bar starting at x=3.75
])
def test_key_centers_match_hand_measured_ansi_stagger(key, expected):
    assert layout.key_positions()[key] == pytest.approx(expected)


@pytest.mark.parametrize("a,b,expected", [
    ("e", "d", "same_finger"),   # both left middle
    ("t", "h", "alternate"),     # left index -> right index
    ("e", "r", "same_hand"),     # left middle -> left index
    ("s", "s", "repeat"),
    (" ", "a", "alternate"),     # thumb counts as its own hand
])
def test_bigram_classes_hand_checked(a, b, expected):
    assert layout.bigram_class(a, b) == expected


def test_truth_coords_follows_KEYS_order():
    coords = layout.truth_coords()
    assert coords.shape == (27, 2)
    pos = layout.key_positions()
    assert np.allclose(coords[0], pos["a"])   # KEYS is alphabetical, then space
    assert np.allclose(coords[-1], pos[" "])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_interp_layout.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'typeshi.interp'`

- [ ] **Step 3: Write the implementation**

```python
# src/typeshi/interp/__init__.py
"""Interpretability probes. Read-only with respect to training and eval."""
```

```python
# src/typeshi/interp/layout.py
"""Physical QWERTY ground truth: where the keys sit, and which finger types them.

Pure data -- no model, no corpus. Every reconstruction metric is scored against
this file, so a wrong entry here silently invalidates the whole result. That is
why the tests hand-check individual keys instead of round-tripping the tables
against themselves.
"""

from __future__ import annotations

import numpy as np

# Canonical key order: 26 lowercase letters then space. Every matrix in this
# package is indexed by this tuple, in this order.
KEYS: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz") + (" ",)

# Standard ANSI stagger in key-width units (1u = one key), y increasing upward.
# The x offsets are the real ones: Tab is 1.5u wide so Q's center lands at 2.0,
# Caps is 1.75u so A lands at 2.25, LShift is 2.25u so Z lands at 2.75.
_ROWS = (
    ("qwertyuiop", 2.00, 1.0),
    ("asdfghjkl", 2.25, 0.0),
    ("zxcvbnm", 2.75, -1.0),
)
# The space bar is 6.25u starting at x=3.75, so its center is 3.75 + 6.25/2.
_SPACE_POS = (6.875, -2.0)

# Standard touch-typing assignment; pinky=4 ... index=1, thumb=0. The thumb is
# its own "hand": calling space left- or right-handed would make every space
# bigram same-hand for one half of the board and alternate for the other, which
# is not how a thumb behaves.
_FINGERS = {
    ("L", 4): "qaz",
    ("L", 3): "wsx",
    ("L", 2): "edc",
    ("L", 1): "rfvtgb",
    ("R", 1): "yhnujm",
    ("R", 2): "ik",
    ("R", 3): "ol",
    ("R", 4): "p",
    ("T", 0): " ",
}


def key_positions() -> dict[str, tuple[float, float]]:
    """Key center coordinates in key-width units."""
    pos: dict[str, tuple[float, float]] = {}
    for chars, x0, y in _ROWS:
        for i, c in enumerate(chars):
            pos[c] = (x0 + float(i), y)
    pos[" "] = _SPACE_POS
    return pos


def truth_coords() -> np.ndarray:
    """(27, 2) ground-truth coordinates in KEYS order."""
    pos = key_positions()
    return np.array([pos[k] for k in KEYS], dtype=float)


def finger_of() -> dict[str, tuple[str, int]]:
    """key -> (hand, finger index)."""
    return {c: hand_finger for hand_finger, chars in _FINGERS.items() for c in chars}


def bigram_class(a: str, b: str) -> str:
    """One of: repeat, same_finger, same_hand, alternate.

    `repeat` is separate from `same_finger` deliberately -- a double tap is one
    of the FASTEST digraphs while a same-finger move is the slowest, so folding
    them together would cancel the very effect this probe is looking for.
    """
    if a == b:
        return "repeat"
    fingers = finger_of()
    (hand_a, fin_a), (hand_b, fin_b) = fingers[a], fingers[b]
    if hand_a != hand_b:
        return "alternate"
    return "same_finger" if fin_a == fin_b else "same_hand"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_interp_layout.py -v`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/interp/__init__.py src/typeshi/interp/layout.py tests/test_interp_layout.py
git commit -m "feat: physical QWERTY ground truth for the interp probes"
```

---

### Task 2: Reconstruction pipeline and the synthetic recovery test

This task lands **before any model is loaded**, on purpose. Recovering a known keyboard from a synthetic latency matrix is what makes a later null result attributable to the model rather than to the analysis.

**Files:**
- Create: `src/typeshi/interp/reconstruct.py`
- Test: `tests/test_interp_reconstruct.py`

**Interfaces:**
- Consumes: `layout.KEYS`, `layout.truth_coords()`, `layout.bigram_class`.
- Produces: `to_log_symmetric(latency_ms) -> np.ndarray`, `median_polish(m, iters=25) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]`, `embed_2d(residual, seed=0) -> np.ndarray` shape `(27, 2)`, `align(coords, truth) -> tuple[np.ndarray, float]`, `score(fitted, truth) -> dict[str, float]`, `permutation_p(coords, truth, n=500, seed=0) -> float`, `synthetic_latency(seed=0, same_finger_penalty=0.35, noise=0.02) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interp_reconstruct.py
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


def test_median_polish_strips_planted_row_and_column_effects():
    rng = np.random.default_rng(0)
    interaction = rng.normal(0, 0.05, (27, 27))
    interaction = (interaction + interaction.T) / 2
    row = rng.normal(0, 0.5, 27)
    planted = interaction + row[:, None] + row[None, :]
    np.fill_diagonal(planted, np.nan)
    residual, row_eff, _, _ = reconstruct.median_polish(planted)
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
    residual, _, _, _ = reconstruct.median_polish(sym)
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


def test_permutation_test_rejects_a_random_labelling():
    truth = layout.truth_coords()
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 3.0, truth.shape)
    assert reconstruct.permutation_p(noise, truth, n=200, seed=0) > 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_interp_reconstruct.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'typeshi.interp.reconstruct'`

- [ ] **Step 3: Write the implementation**

```python
# src/typeshi/interp/reconstruct.py
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


def median_polish(m: np.ndarray, iters: int = 25):
    """Tukey median polish -> (residual, row_effect, col_effect, grand).

    Load-bearing. Some keys are simply slow, and without stripping row and
    column effects the leading component of the residual is "fast keys vs slow
    keys" and geometry never surfaces. NaN-aware, so the masked diagonal and
    any masked thin cells cannot poison a median. Median rather than mean
    because same-finger outliers are exactly what we do not want setting the
    baseline.
    """
    r = np.array(m, dtype=float, copy=True)
    row = np.zeros(r.shape[0])
    col = np.zeros(r.shape[1])
    grand = 0.0
    for _ in range(iters):
        rm = np.nanmedian(r, axis=1)
        r -= rm[:, None]
        row += rm
        cm = np.nanmedian(r, axis=0)
        r -= cm[None, :]
        col += cm
        g = np.median(row)
        row -= g
        grand += g
        g = np.median(col)
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
        dissimilarity="precomputed",
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


def score(fitted: np.ndarray, truth: np.ndarray, keys=layout.KEYS) -> dict:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_interp_reconstruct.py -v`
Expected: PASS, 4 tests. Record the actual `distance_spearman` from the synthetic test — it is the pipeline's ceiling and every model number is read against it. If it comes in below 0.85, the pipeline has a bug; debug it rather than lowering the threshold.

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/interp/reconstruct.py tests/test_interp_reconstruct.py
git commit -m "feat: reconstruction pipeline, validated on a synthetic keyboard"
```

---

### Task 3: Same-finger handling, blind and finger-aware

The blind detector is what keeps the headline claim non-circular: it finds the same-finger pairs from the residual distribution alone, and that detection is then *scored* against truth rather than assumed.

**Files:**
- Modify: `src/typeshi/interp/reconstruct.py` (append)
- Test: `tests/test_interp_reconstruct.py` (append)

**Interfaces:**
- Consumes: `median_polish`, `remove_indicator`, `embed_2d`, `align`, `score` from Task 2.
- Produces: `detect_same_finger(residual) -> np.ndarray` (bool, symmetric), `true_same_finger(keys=layout.KEYS) -> np.ndarray`, `same_finger_auc(residual, truth_mask) -> float`, `reconstruct(latency_ms, mode="blind", seed=0) -> dict` with keys `fitted`, `residual`, `detected_mask`, `metrics`, `same_finger_auc`, `disparity`, `mode`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_interp_reconstruct.py

def test_blind_detector_finds_planted_same_finger_pairs():
    # No ground truth reaches the detector -- it sees only the residual.
    latency = reconstruct.synthetic_latency(seed=0)
    sym = reconstruct.to_log_symmetric(latency)
    residual, _, _, _ = reconstruct.median_polish(sym)
    truth_mask = reconstruct.true_same_finger()
    assert reconstruct.same_finger_auc(residual, truth_mask) > 0.9
    detected = reconstruct.detect_same_finger(residual)
    assert detected.shape == (27, 27)
    assert (detected == detected.T).all()   # symmetric, like the residual


def test_both_modes_reconstruct_the_synthetic_keyboard():
    latency = reconstruct.synthetic_latency(seed=0)
    blind = reconstruct.reconstruct(latency, mode="blind", seed=0)
    aware = reconstruct.reconstruct(latency, mode="finger_aware", seed=0)
    assert blind["metrics"]["distance_spearman"] > 0.80
    assert aware["metrics"]["distance_spearman"] > 0.85
    # Ground truth can only help; if blind beats aware something is wrong.
    assert aware["metrics"]["distance_spearman"] >= blind["metrics"]["distance_spearman"] - 0.1


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        reconstruct.reconstruct(reconstruct.synthetic_latency(), mode="magic")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_interp_reconstruct.py -v -k "blind or both_modes or unknown_mode"`
Expected: FAIL — `AttributeError: module 'typeshi.interp.reconstruct' has no attribute 'true_same_finger'`

- [ ] **Step 3: Write the implementation**

```python
# append to src/typeshi/interp/reconstruct.py

def true_same_finger(keys=layout.KEYS) -> np.ndarray:
    """Ground-truth same-finger mask. Diagnostic and scoring use only."""
    mask = np.zeros((len(keys), len(keys)), dtype=bool)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            mask[i, j] = layout.bigram_class(a, b) == "same_finger"
    return mask


def detect_same_finger(residual: np.ndarray) -> np.ndarray:
    """Blind same-finger mask: the slow mode of a two-component mixture.

    Uses no ground truth. Same-finger moves are the slow outliers of the
    interaction residual, and a two-component Gaussian mixture separates them
    without being told how many there are -- which is what keeps the blind
    reconstruction non-circular. Thresholding at a percentile would smuggle the
    true count (41 of 351 unordered pairs) straight back in.
    """
    from sklearn.mixture import GaussianMixture

    upper = np.triu_indices(residual.shape[0], k=1)
    vals = residual[upper]
    ok = ~np.isnan(vals)
    gm = GaussianMixture(n_components=2, random_state=0).fit(
        vals[ok].reshape(-1, 1)
    )
    slow = int(np.argmax(gm.means_.ravel()))
    flags = np.zeros(vals.shape, dtype=bool)
    flags[ok] = gm.predict(vals[ok].reshape(-1, 1)) == slow
    mask = np.zeros_like(residual, dtype=bool)
    mask[upper] = flags
    return mask | mask.T


def same_finger_auc(residual: np.ndarray, truth_mask: np.ndarray) -> float:
    """How well the raw residual ranks true same-finger pairs to the top.

    Reported alongside the blind reconstruction: it is the evidence that the
    detector had something real to find, independent of where MDS then puts
    the keys.
    """
    from sklearn.metrics import roc_auc_score

    upper = np.triu_indices(residual.shape[0], k=1)
    vals, labels = residual[upper], truth_mask[upper]
    ok = ~np.isnan(vals)
    return float(roc_auc_score(labels[ok], vals[ok]))


def reconstruct(latency_ms: np.ndarray, mode: str = "blind", seed: int = 0,
                keys=layout.KEYS) -> dict:
    """Full pipeline: latencies -> scored 2D layout.

    `mode="blind"` uses no ground truth anywhere and is the honest
    from-scratch number. `mode="finger_aware"` regresses out the TRUE
    same-finger indicator and is diagnostic only -- it answers "is the geometry
    there once the biomechanical confound is accounted for" and must be
    labelled as ground-truth-assisted wherever it is reported.
    """
    if mode not in ("blind", "finger_aware"):
        raise ValueError(f"unknown mode {mode!r}; expected blind or finger_aware")
    truth = layout.truth_coords()
    residual, _, _, _ = median_polish(to_log_symmetric(latency_ms))
    truth_mask = true_same_finger(keys)
    detected = detect_same_finger(residual)
    mask = detected if mode == "blind" else truth_mask
    adjusted = remove_indicator(residual, mask)
    coords = embed_2d(adjusted, seed=seed)
    fitted, disparity = align(coords, truth)
    return {
        "mode": mode,
        "fitted": fitted,
        "residual": residual,
        "detected_mask": detected,
        "same_finger_auc": same_finger_auc(residual, truth_mask),
        "disparity": disparity,
        "metrics": score(fitted, truth, keys),
    }
```

- [ ] **Step 4: Run the full test file**

Run: `uv run pytest tests/test_interp_reconstruct.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/interp/reconstruct.py tests/test_interp_reconstruct.py
git commit -m "feat: blind and finger-aware same-finger handling"
```

---

### Task 4: Corpus statistics

The data ceiling. If the corpus does not encode QWERTY, the model cannot. One pass produces everything the probe and the controls need.

**Files:**
- Create: `src/typeshi/interp/empirical.py`
- Create: `tests/fixtures/interp_digraph.jsonl`
- Test: `tests/test_interp_empirical.py`

**Interfaces:**
- Consumes: `layout.KEYS`, `typeshi.serialize.deserialize`, `typeshi.events.EventType`, `typeshi.timebins.to_bin`, `typeshi.config.TIME_BINS`.
- Produces: `corpus_stats(path, limit=200_000, total_lines=1_975_019, seed=0, keys=layout.KEYS) -> CorpusStats`, a dataclass with fields `latency_ms: np.ndarray (27,27)`, `counts: np.ndarray (27,27) int`, `modal_hold: dict[str, int]`, `modal_dt: np.ndarray (27,27) int`, `global_modal_dt: int`, `bigram_counts: np.ndarray (27,27) int`.

- [ ] **Step 1: Write the fixture and the failing test**

```jsonl
{"prompt": "<MODE:T><WPM:10><ECOR:0><EUNC:0><REV:0><TARGET>the<PROCESS>", "completion": "<t:50><DT:60><h:50><DT:61><e:52>"}
{"prompt": "<MODE:T><WPM:10><ECOR:0><EUNC:0><REV:0><TARGET>ab<PROCESS>", "completion": "<a:50><DT:60><BKSP:50><DT:61><b:50>"}
{"prompt": "<MODE:T><WPM:10><ECOR:0><EUNC:0><REV:0><TARGET>t he<PROCESS>", "completion": "<t:50><DT:60><SPC:55><DT:62><h:50>"}
```

```python
# tests/test_interp_empirical.py
from pathlib import Path

import numpy as np
import pytest
from typeshi.interp import empirical, layout
from typeshi.timebins import from_bin

FIXTURE = Path(__file__).parent / "fixtures" / "interp_digraph.jsonl"
INDEX = {k: i for i, k in enumerate(layout.KEYS)}


@pytest.fixture(scope="module")
def stats():
    # total_lines equals the fixture length so the sampler keeps every line.
    return empirical.corpus_stats(FIXTURE, limit=10, total_lines=3, seed=0)


def test_adjacent_key_pairs_are_measured_at_their_bin_center(stats):
    th = stats.latency_ms[INDEX["t"], INDEX["h"]]
    assert th == pytest.approx(from_bin(60))
    assert stats.counts[INDEX["t"], INDEX["h"]] == 1


def test_pairs_bridged_by_a_backspace_are_skipped(stats):
    # <a:50><DT:60><BKSP:50><DT:61><b:50> is NOT an a->b digraph: the finger
    # went somewhere else in between, so its latency says nothing about a->b.
    assert stats.counts[INDEX["a"], INDEX["b"]] == 0
    assert np.isnan(stats.latency_ms[INDEX["a"], INDEX["b"]])


def test_space_is_a_first_class_key(stats):
    assert stats.counts[INDEX["t"], INDEX[" "]] == 1
    assert stats.latency_ms[INDEX["t"], INDEX[" "]] == pytest.approx(from_bin(60))


def test_modal_hold_and_dt_come_back_as_bin_indices(stats):
    assert stats.modal_hold["t"] == 50
    assert stats.modal_hold[" "] == 55
    assert stats.modal_dt[INDEX["t"], INDEX["h"]] == 60
    assert 0 <= stats.global_modal_dt < 128


def test_target_bigram_counts_come_from_the_prompt_text(stats):
    # Used only by the frequency control, so it must count TARGET text --
    # not the event stream, which is what we are trying to explain.
    assert stats.bigram_counts[INDEX["t"], INDEX["h"]] == 1   # "the" only
    assert stats.bigram_counts[INDEX["t"], INDEX[" "]] == 1   # "t he"
    assert stats.bigram_counts[INDEX[" "], INDEX["h"]] == 1   # "t he"


def test_median_is_computed_alongside_the_geometric_mean(stats):
    # Spec §3.3 keeps the median as a robustness check: the geometric mean is
    # the functional the model readout uses, but a few multi-second thinking
    # pauses in a thin cell can still drag it.
    assert stats.median_ms[INDEX["t"], INDEX["h"]] == pytest.approx(from_bin(60))
    assert np.isnan(stats.median_ms[INDEX["a"], INDEX["b"]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_interp_empirical.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'typeshi.interp.empirical'`

- [ ] **Step 3: Write the implementation**

```python
# src/typeshi/interp/empirical.py
"""Digraph statistics measured straight from the corpus.

This is the ceiling the model is read against: if the data does not encode
QWERTY, no model trained on it can. It is also where the probe gets its
realistic hold and gap bins, so the carrier prefix is corpus-shaped rather
than invented.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from typeshi import config
from typeshi.events import EventType
from typeshi.interp import layout
from typeshi.serialize import deserialize
from typeshi.timebins import to_bin

_TARGET_RE = re.compile(r"<TARGET>(.*)<PROCESS>", re.DOTALL)


@dataclass
class CorpusStats:
    latency_ms: np.ndarray      # (27, 27) geometric-mean press-to-press ms
    median_ms: np.ndarray       # (27, 27) median, the robustness check
    counts: np.ndarray          # (27, 27) support per cell
    modal_hold: dict[str, int]  # key -> most common hold bin
    modal_dt: np.ndarray        # (27, 27) most common gap bin
    global_modal_dt: int
    bigram_counts: np.ndarray   # (27, 27) counts in TARGET text


def _histogram_median(hist: np.ndarray) -> np.ndarray:
    """Per-cell median bin center from a (n, n, TIME_BINS) count histogram.

    Read off the histogram rather than kept as a list of samples: the corpus
    pass sees tens of millions of digraphs and holding them all to sort later
    would cost gigabytes for a single robustness column.
    """
    from typeshi.timebins import from_bin

    totals = hist.sum(axis=2)
    cumulative = np.cumsum(hist, axis=2)
    out = np.full(totals.shape, np.nan)
    rows, cols = np.nonzero(totals)
    for i, j in zip(rows, cols):
        bin_index = int(np.searchsorted(cumulative[i, j], totals[i, j] / 2.0))
        out[i, j] = from_bin(min(bin_index, config.TIME_BINS - 1))
    return out


def corpus_stats(path: Path | str, limit: int = 200_000,
                 total_lines: int = 1_975_019, seed: int = 0,
                 keys=layout.KEYS) -> CorpusStats:
    """One pass over a seeded Bernoulli sample of `path`.

    Bernoulli rather than reservoir sampling: train.jsonl is ~1.4 GB and
    ordered by writer, so a head-of-file slice would cover a handful of
    typists. Sampling with p = limit/total_lines streams in one pass and keeps
    the full writer breadth.

    Only lowercase letters and space are counted. Uppercase is dropped rather
    than folded in: a capital costs a shift chord, which is a different motor
    event, and merging it would blur the very finger structure being measured.
    """
    index = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    log_sum = np.zeros((n, n))
    counts = np.zeros((n, n), dtype=int)
    dt_hist = np.zeros((n, n, config.TIME_BINS), dtype=int)
    hold_hist = np.zeros((n, config.TIME_BINS), dtype=int)
    bigram_counts = np.zeros((n, n), dtype=int)

    rng = random.Random(seed)
    keep_p = min(1.0, limit / max(total_lines, 1))
    with open(path) as handle:
        for line in handle:
            if rng.random() >= keep_p:
                continue
            record = json.loads(line)
            events = deserialize(record["completion"])
            for event in events:
                if event.type is EventType.KEY and event.char in index:
                    hold = to_bin(event.release_time - event.press_time)
                    hold_hist[index[event.char], hold] += 1
            for prev, cur in zip(events, events[1:]):
                if prev.type is not EventType.KEY or cur.type is not EventType.KEY:
                    continue
                if prev.char not in index or cur.char not in index:
                    continue
                gap = cur.press_time - prev.press_time
                if gap <= 0:
                    continue
                i, j = index[prev.char], index[cur.char]
                log_sum[i, j] += math.log(gap)
                counts[i, j] += 1
                dt_hist[i, j, to_bin(gap)] += 1
            match = _TARGET_RE.search(record["prompt"])
            if match:
                text = match.group(1)
                for a, b in zip(text, text[1:]):
                    if a in index and b in index:
                        bigram_counts[index[a], index[b]] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        latency = np.exp(log_sum / counts)
    total_dt = dt_hist.sum(axis=(0, 1))
    return CorpusStats(
        latency_ms=latency,
        median_ms=_histogram_median(dt_hist),
        counts=counts,
        modal_hold={
            k: int(np.argmax(hold_hist[i])) for i, k in enumerate(keys)
            if hold_hist[i].sum() > 0
        },
        modal_dt=dt_hist.argmax(axis=2),
        global_modal_dt=int(np.argmax(total_dt)) if total_dt.sum() else 0,
        bigram_counts=bigram_counts,
    )


def mask_thin_cells(stats: CorpusStats, min_support: int = 20) -> np.ndarray:
    """Latencies with under-supported cells set to NaN.

    Masked rather than kept: a cell backed by three observations is noise, and
    median polish plus MDS will happily treat that noise as geometry.
    """
    out = np.array(stats.latency_ms, dtype=float, copy=True)
    out[stats.counts < min_support] = np.nan
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_interp_empirical.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Sanity-run against the real corpus**

Run:
```bash
uv run python -c "
from typeshi.interp import empirical, layout, reconstruct
import numpy as np
s = empirical.corpus_stats('data/processed/train.jsonl', limit=50_000)
print('cells with >=20 support:', int((s.counts >= 20).sum()), 'of 729')
res, _, _, _ = reconstruct.median_polish(reconstruct.to_log_symmetric(empirical.mask_thin_cells(s)))
print('same-finger AUC from data alone:', reconstruct.same_finger_auc(res, reconstruct.true_same_finger()))
"
```
Expected: a coverage count and an AUC. This is the first real evidence in the project that corpus timing carries finger structure — record the number in the task's commit message.

- [ ] **Step 6: Commit**

```bash
git add src/typeshi/interp/empirical.py tests/test_interp_empirical.py tests/fixtures/interp_digraph.jsonl
git commit -m "feat: corpus digraph statistics and the data-side ceiling"
```

---

### Task 5: The model probe

**Files:**
- Modify: `src/typeshi/serialize.py` (add a public `encode_char`)
- Create: `src/typeshi/interp/digraph.py`
- Test: `tests/test_interp_digraph.py`

**Interfaces:**
- Consumes: `empirical.CorpusStats`, `layout.KEYS`, `typeshi.dataset.build_prompt`, `typeshi.labels.SessionLabels`, `typeshi.timebins.from_bin`, `typeshi.config.TIME_BINS`.
- Produces: `CARRIER_HEAD: str`, `CARRIER_TAIL: str`, `carrier_target(a, b) -> str`, `probe_labels(wpm=52.5) -> SessionLabels`, `dt_token_ids(tok) -> list[int]`, `build_prefix(modal_hold, modal_dt, global_modal_dt) -> str`, `probe_matrix(model, tok, stats, wpm=52.5, hold_override=None, batch_size=64, keys=layout.KEYS) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interp_digraph.py
import numpy as np
import pytest
from typeshi.interp import digraph, layout
from typeshi.tiny_tokenizer import build_tiny_tokenizer


@pytest.fixture(scope="module")
def tok():
    return build_tiny_tokenizer()


def test_carrier_puts_the_bigram_after_a_letter():
    # A "... of {a}{b} ..." frame would make every space-initial cell type a
    # DOUBLE space -- a different motor event from the other 26 rows.
    assert digraph.carrier_target("q", "z") == "the ratiosqz in the sample"
    assert digraph.carrier_target(" ", "a") == "the ratios a in the sample"
    assert digraph.CARRIER_HEAD[-1].isalpha()


def test_every_cell_has_the_same_token_length(tok):
    # Equal lengths are what let all 729 probes batch without padding, and
    # they are also the invariant that says only the bigram varies.
    lengths = {
        len(tok(digraph.carrier_target(a, b), add_special_tokens=False)["input_ids"])
        for a in ("q", "z", " ") for b in ("m", " ", "e")
    }
    assert len(lengths) == 1


def test_dt_token_ids_are_complete_and_distinct(tok):
    ids = digraph.dt_token_ids(tok)
    assert len(ids) == 128
    assert len(set(ids)) == 128
    assert all(isinstance(i, int) for i in ids)


def test_prefix_is_byte_identical_and_types_the_carrier_head(tok):
    from typeshi.serialize import deserialize

    modal_hold = {c: 50 for c in layout.KEYS}
    modal_dt = np.full((27, 27), 60, dtype=int)
    prefix = digraph.build_prefix(modal_hold, modal_dt, 60)
    events = deserialize(prefix)
    assert "".join(e.char for e in events) == digraph.CARRIER_HEAD
    # Constructed, not decoded: identical every call, so no prompt-dependent
    # drift can leak into the cells the probe is supposed to isolate.
    assert prefix == digraph.build_prefix(modal_hold, modal_dt, 60)


def test_probe_matrix_reads_the_dt_softmax(tok):
    """A stub model whose DT logits encode a known bin must come back as that
    bin's center -- this pins the readout, not the model."""
    import torch
    from typeshi.timebins import from_bin

    target_bin = 60
    dt_ids = digraph.dt_token_ids(tok)

    class StubModel:
        device = "cpu"

        def __call__(self, input_ids, **kwargs):
            logits = torch.full(
                (input_ids.shape[0], input_ids.shape[1], len(tok)), -1e4
            )
            logits[:, -1, dt_ids[target_bin]] = 1e4
            return type("Out", (), {"logits": logits})()

        def eval(self):
            return self

    stats = type("S", (), {
        "modal_hold": {c: 50 for c in layout.KEYS},
        "modal_dt": np.full((27, 27), 60, dtype=int),
        "global_modal_dt": 60,
    })()
    matrix = digraph.probe_matrix(StubModel(), tok, stats, batch_size=16)
    assert matrix.shape == (27, 27)
    assert matrix[0, 1] == pytest.approx(from_bin(target_bin), rel=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_interp_digraph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'typeshi.interp.digraph'`

- [ ] **Step 3: Add the public helper to serialize.py**

Insert directly after the existing `_decode_char` definition (`src/typeshi/serialize.py:66-67`):

```python
def encode_char(c: str) -> str:
    """Public alias for the escape table: 'a' -> 'a', ' ' -> 'SPC'.

    The interp probes build event tokens by hand and need the same escaping
    serialize() uses; reaching into _encode_char from another module would
    make a private detail load-bearing.
    """
    return _encode_char(c)
```

- [ ] **Step 4: Write the probe**

```python
# src/typeshi/interp/digraph.py
"""Reads the model's beliefs about digraph cost straight off the <DT:> softmax.

No trained probe is needed: the 128-way <DT:k> distribution IS the model's
belief about how long the next gap will be, and teacher-forcing a fixed carrier
up to a chosen keypress exposes it directly. Every cell of the resulting matrix
differs only in the two characters of the bigram -- the controlled contrast a
model can give you and an observational corpus cannot.
"""

from __future__ import annotations

import numpy as np

from typeshi import config
from typeshi.dataset import build_prompt
from typeshi.interp import layout
from typeshi.labels import SessionLabels
from typeshi.serialize import encode_char
from typeshi.timebins import from_bin

# The insertion point follows a LETTER, not a space: with a "... of {a}{b} ..."
# frame the 27 cells where {a} is space would type a double space and read as a
# different motor event from every other row.
CARRIER_HEAD = "the ratios"
CARRIER_TAIL = " in the sample"


def carrier_target(a: str, b: str) -> str:
    return f"{CARRIER_HEAD}{a}{b}{CARRIER_TAIL}"


def probe_labels(wpm: float = 52.5) -> SessionLabels:
    """Fixed conditioning for every cell. 52.5 wpm lands in bin 10, near the
    corpus mode; error and revision knobs are zeroed so no cell is nudged
    toward emitting a correction instead of the next key."""
    return SessionLabels(
        wpm=wpm,
        corrected_error_rate=0.0,
        uncorrected_error_rate=0.0,
        revision_rate=0.0,
    )


def dt_token_ids(tok) -> list[int]:
    """The 128 <DT:k> IDs, in bin order."""
    ids = [tok.convert_tokens_to_ids(f"<DT:{k}>") for k in range(config.TIME_BINS)]
    if any(i is None for i in ids) or len(set(ids)) != config.TIME_BINS:
        raise ValueError("tokenizer does not carry all 128 <DT:> tokens as single IDs")
    return [int(i) for i in ids]


def build_prefix(modal_hold: dict[str, int], modal_dt: np.ndarray,
                 global_modal_dt: int, keys=layout.KEYS) -> str:
    """The event stream for typing CARRIER_HEAD, built deterministically.

    Constructed from corpus modal bins rather than decoded from the model.
    Decoding would condition on the prompt, whose bigram differs per cell, so
    the prefix would vary by more than the two characters the probe isolates --
    and it would silently change whenever the checkpoint did. Construction is
    byte-identical by definition and needs no forward pass.
    """
    index = {k: i for i, k in enumerate(keys)}
    parts: list[str] = []
    for position, char in enumerate(CARRIER_HEAD):
        if position:
            previous = CARRIER_HEAD[position - 1]
            gap = global_modal_dt
            if previous in index and char in index:
                gap = int(modal_dt[index[previous], index[char]]) or global_modal_dt
            parts.append(f"<DT:{gap}>")
        parts.append(f"<{encode_char(char)}:{modal_hold.get(char, 50)}>")
    return "".join(parts)


def probe_matrix(model, tok, stats, wpm: float = 52.5,
                 hold_override: int | None = None, batch_size: int = 64,
                 keys=layout.KEYS) -> np.ndarray:
    """(27, 27) predicted press-to-press latencies in ms, ordered pairs.

    `hold_override` forces one hold bin for every {a} press instead of the
    per-character corpus mode -- the sensitivity check the spec requires.
    """
    import torch

    labels = probe_labels(wpm)
    prefix = build_prefix(stats.modal_hold, stats.modal_dt, stats.global_modal_dt, keys)
    dt_ids = torch.tensor(dt_token_ids(tok))
    log_centers = torch.tensor(
        [float(np.log(from_bin(k))) for k in range(config.TIME_BINS)],
        dtype=torch.float32,
    )

    sequences: list[list[int]] = []
    for a in keys:
        hold = hold_override if hold_override is not None else stats.modal_hold.get(a, 50)
        for b in keys:
            text = build_prompt(carrier_target(a, b), labels, "transcription")
            text += prefix + f"<{encode_char(a)}:{hold}>"
            sequences.append(tok(text, add_special_tokens=False)["input_ids"])

    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(
            f"probe sequences must be equal length for an unpadded batch; got {lengths}"
        )

    out = np.zeros(len(keys) * len(keys))
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            chunk = torch.tensor(sequences[start:start + batch_size])
            logits = model(input_ids=chunk.to(model.device)).logits[:, -1, :].float().cpu()
            # Restrict to the DT block and renormalize: the grammar guarantees a
            # gap token comes next, so mass anywhere else is not a competing
            # prediction, it is off-manifold noise.
            probs = torch.softmax(logits[:, dt_ids], dim=-1)
            # Geometric mean: the bins are geomspaced, so an arithmetic mean
            # over bin centers would be dominated by the long-pause tail.
            out[start:start + chunk.shape[0]] = torch.exp(
                (probs * log_centers).sum(dim=-1)
            ).numpy()
    return out.reshape(len(keys), len(keys))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_interp_digraph.py tests/test_serialize.py -v`
Expected: PASS — the new file plus no regression in serialize

- [ ] **Step 6: Commit**

```bash
git add src/typeshi/serialize.py src/typeshi/interp/digraph.py tests/test_interp_digraph.py
git commit -m "feat: read digraph latencies off the model's <DT:> softmax"
```

---

### Task 6: The probe script, controls, and report

**Files:**
- Create: `scripts/keyboard_probe.py`
- Modify: `pyproject.toml` (add the `viz` extra)
- Test: `tests/test_keyboard_probe.py`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `frequency_control(latency_ms, bigram_counts) -> np.ndarray`, `biomechanical_table(latency_ms, keys=layout.KEYS) -> dict[str, float]`, `random_init_model(checkpoint) -> model`, `run(...) -> dict`, and a CLI writing `keyboard_probe_<name>.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_keyboard_probe.py
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from typeshi.interp import layout, reconstruct

spec = importlib.util.spec_from_file_location(
    "keyboard_probe", Path(__file__).parent.parent / "scripts" / "keyboard_probe.py"
)
kp = importlib.util.module_from_spec(spec)
sys.modules["keyboard_probe"] = kp
spec.loader.exec_module(kp)


def test_biomechanical_ordering_on_the_synthetic_matrix():
    # The level-1 result: alternate < same_hand < same_finger, repeat fastest.
    # Robust to MDS failing entirely, which is why it is the headline.
    table = kp.biomechanical_table(reconstruct.synthetic_latency(seed=0))
    assert table["repeat"] < table["alternate"] < table["same_hand"] < table["same_finger"]


def test_frequency_control_removes_a_planted_frequency_effect():
    # If layout only appeared because common bigrams are typed fast, stripping
    # log-count would erase the structure. Here the effect is PURE frequency,
    # so the residual must come out flat -- any leftover signal is a bug.
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10_000, (27, 27))
    latency = np.exp(6.0 - 0.1 * np.log(counts + 1.0))
    controlled = kp.frequency_control(latency, counts)
    assert np.nanstd(np.log(controlled)) < 0.1 * np.nanstd(np.log(latency))


def test_report_carries_every_scoring_level_and_labels_ground_truth_use():
    report = kp.build_report(
        model_latency=reconstruct.synthetic_latency(seed=0),
        empirical_latency=reconstruct.synthetic_latency(seed=1),
        random_latency=reconstruct.synthetic_latency(seed=2),
        bigram_counts=np.ones((27, 27), dtype=int),
        seed=0,
    )
    assert set(report["model"]) >= {"biomechanical", "blind", "finger_aware",
                                    "same_finger_auc", "frequency_controlled"}
    assert report["model"]["finger_aware"]["uses_ground_truth"] is True
    assert report["model"]["blind"]["uses_ground_truth"] is False
    assert "empirical" in report and "random_init" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_keyboard_probe.py -v`
Expected: FAIL — `FileNotFoundError: scripts/keyboard_probe.py`

- [ ] **Step 3: Add the viz extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:

```toml
viz = ["matplotlib>=3.8"]
```

- [ ] **Step 4: Write the script**

```python
# scripts/keyboard_probe.py
"""Recover a QWERTY layout from the tiny motor model and score it.

    uv run python scripts/keyboard_probe.py --checkpoint checkpoints/interp-snapshots/step-15000

CPU by default, deliberately: a training run owns the MPS device and a probe
that steals it slows the real work (same reasoning as scripts/playground.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from typeshi.interp import digraph, empirical, layout, reconstruct


def biomechanical_table(latency_ms: np.ndarray, keys=layout.KEYS) -> dict[str, float]:
    """Mean latency in ms by bigram class -- the level-1 result.

    Needs no MDS and no alignment, so it survives a completely failed
    reconstruction. If the ordering is right the model has learned motor
    structure even when the 2D map looks like soup.
    """
    buckets: dict[str, list[float]] = {}
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            value = latency_ms[i, j]
            if np.isnan(value):
                continue
            buckets.setdefault(layout.bigram_class(a, b), []).append(float(value))
    return {name: float(np.mean(values)) for name, values in buckets.items()}


def frequency_control(latency_ms: np.ndarray, bigram_counts: np.ndarray) -> np.ndarray:
    """Latencies with the log-bigram-frequency trend regressed out.

    Rules out the deflationary reading: that the "keyboard" is really letter
    statistics, since common bigrams are both fast and English-shaped.
    """
    log_latency = np.log(np.asarray(latency_ms, dtype=float))
    log_count = np.log(np.asarray(bigram_counts, dtype=float) + 1.0)
    ok = ~np.isnan(log_latency)
    slope, intercept = np.polyfit(log_count[ok], log_latency[ok], 1)
    residual = log_latency - (slope * log_count + intercept)
    # Re-center on the original mean so the output is still interpretable as ms.
    return np.exp(residual + np.nanmean(log_latency))


def scored(latency: np.ndarray, mode: str, seed: int) -> dict:
    result = reconstruct.reconstruct(latency, mode=mode, seed=seed)
    return {
        "uses_ground_truth": mode == "finger_aware",
        "metrics": result["metrics"],
        "disparity": result["disparity"],
        "permutation_p": reconstruct.permutation_p(
            result["fitted"], layout.truth_coords(), n=500, seed=seed
        ),
        "coords": result["fitted"].tolist(),
    }


def analyze(latency: np.ndarray, bigram_counts: np.ndarray, seed: int) -> dict:
    residual, _, _, _ = reconstruct.median_polish(reconstruct.to_log_symmetric(latency))
    return {
        "biomechanical": biomechanical_table(latency),
        "same_finger_auc": reconstruct.same_finger_auc(
            residual, reconstruct.true_same_finger()
        ),
        "blind": scored(latency, "blind", seed),
        "finger_aware": scored(latency, "finger_aware", seed),
        "frequency_controlled": scored(
            frequency_control(latency, bigram_counts), "blind", seed
        ),
    }


def build_report(model_latency, empirical_latency, random_latency,
                 bigram_counts, seed: int = 0) -> dict:
    return {
        "keys": list(layout.KEYS),
        "model": analyze(model_latency, bigram_counts, seed),
        "empirical": analyze(empirical_latency, bigram_counts, seed),
        "random_init": analyze(random_latency, bigram_counts, seed),
    }


def random_init_model(checkpoint: Path):
    """Same architecture, random weights -- the control that catches a probe or
    an analysis manufacturing structure out of nothing."""
    from transformers import AutoConfig, AutoModelForCausalLM

    return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(checkpoint))


def write_figure(report: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    truth = layout.truth_coords()
    panels = [("model", "blind"), ("model", "finger_aware"), ("empirical", "blind")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (source, mode) in zip(axes, panels):
        fitted = np.array(report[source][mode]["coords"])
        ax.scatter(truth[:, 0], truth[:, 1], s=180, facecolors="none",
                   edgecolors="0.75")
        for key, (x, y) in zip(layout.KEYS, fitted):
            ax.text(x, y, "␣" if key == " " else key, ha="center", va="center")
        rho = report[source][mode]["metrics"]["distance_spearman"]
        ax.set_title(f"{source} / {mode}\nrho={rho:.2f}")
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/interp-snapshots/step-15000"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--limit", type=int, default=200_000)
    parser.add_argument("--wpm", type=float, default=52.5)
    parser.add_argument("--hold-override", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    from typeshi.eval.load import load_checkpoint_tokenizer

    print(f"corpus stats from {args.data} (limit {args.limit})...")
    stats = empirical.corpus_stats(args.data, limit=args.limit, seed=args.seed)
    print(f"  cells with >= {args.min_support} support: "
          f"{int((stats.counts >= args.min_support).sum())} of {len(layout.KEYS) ** 2}")

    tok = load_checkpoint_tokenizer(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(args.device)
    print(f"probing {args.checkpoint}...")
    model_latency = digraph.probe_matrix(
        model, tok, stats, wpm=args.wpm, hold_override=args.hold_override
    )
    print("probing a random-init control...")
    random_latency = digraph.probe_matrix(
        random_init_model(args.checkpoint).to(args.device), tok, stats, wpm=args.wpm
    )

    report = build_report(
        model_latency=model_latency,
        empirical_latency=empirical.mask_thin_cells(stats, args.min_support),
        random_latency=random_latency,
        bigram_counts=stats.bigram_counts,
        seed=args.seed,
    )
    report["checkpoint"] = str(args.checkpoint)
    report["wpm"] = args.wpm
    report["hold_override"] = args.hold_override

    out = args.out or Path(f"keyboard_probe_{args.checkpoint.name}.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")
    for source in ("model", "empirical", "random_init"):
        block = report[source]
        print(f"  {source:12s} blind rho={block['blind']['metrics']['distance_spearman']:+.3f} "
              f"sf_auc={block['same_finger_auc']:.3f} "
              f"hand={block['blind']['metrics']['hand_accuracy']:.3f}")
    if args.figure:
        write_figure(report, args.figure)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_keyboard_probe.py -v`
Expected: PASS, 3 tests

- [ ] **Step 6: Run it for real**

Run:
```bash
uv run pip install -e '.[viz]'
uv run python scripts/keyboard_probe.py \
  --checkpoint checkpoints/interp-snapshots/step-15000 \
  --figure keyboard_step-15000.png
```
Expected: a JSON report and a three-panel figure. The three printed rows are the result: `model` and `empirical` should carry structure, `random_init` should not.

- [ ] **Step 7: Sensitivity check**

Run the same command twice more with `--hold-override` set to the corpus modal hold bin and to the corpus median hold bin (both printed by the corpus-stats step). Record all three `distance_spearman` values. The spec requires reporting the spread, not picking the best.

- [ ] **Step 8: Commit**

```bash
git add scripts/keyboard_probe.py tests/test_keyboard_probe.py pyproject.toml
git commit -m "feat: keyboard probe script with frequency and random-init controls"
```

---

### Task 7: Sweeps

**Files:**
- Create: `scripts/keyboard_sweep.py`
- Test: `tests/test_keyboard_sweep.py`

**Interfaces:**
- Consumes: `keyboard_probe.build_report`, `digraph.probe_matrix`, `empirical.corpus_stats`.
- Produces: `sweep_wpm(model, tok, stats, wpms, seed) -> dict[str, dict]`, `sweep_checkpoints(paths, stats, wpm, device, seed) -> dict[str, dict]`, and a CLI writing `keyboard_sweep.json` plus `keyboard_sweep.png`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_keyboard_sweep.py
import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "keyboard_sweep", Path(__file__).parent.parent / "scripts" / "keyboard_sweep.py"
)
ks = importlib.util.module_from_spec(spec)
sys.modules["keyboard_sweep"] = ks
spec.loader.exec_module(ks)


def test_summarize_pulls_one_row_per_run():
    from typeshi.interp import reconstruct
    import keyboard_probe as kp

    runs = {
        "wpm=25": kp.analyze(reconstruct.synthetic_latency(seed=0),
                              np.ones((27, 27), dtype=int), 0),
        "wpm=80": kp.analyze(reconstruct.synthetic_latency(seed=1),
                              np.ones((27, 27), dtype=int), 0),
    }
    rows = ks.summarize(runs)
    assert [r["run"] for r in rows] == ["wpm=25", "wpm=80"]
    assert all({"distance_spearman", "same_finger_auc", "hand_accuracy"} <= set(r)
               for r in rows)


def test_checkpoint_steps_sort_numerically_not_lexically():
    # step-9000 must come BEFORE step-12000; a lexical sort would invert them
    # and the developmental trajectory would read backwards.
    paths = [Path("x/step-12000"), Path("x/step-9000"), Path("x/step-3000")]
    assert [p.name for p in ks.sorted_snapshots(paths)] == [
        "step-3000", "step-9000", "step-12000"
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_keyboard_sweep.py -v`
Expected: FAIL — `FileNotFoundError: scripts/keyboard_sweep.py`

- [ ] **Step 3: Write the script**

```python
# scripts/keyboard_sweep.py
"""Two sweeps over the keyboard probe: speed conditioning, and training time.

    uv run python scripts/keyboard_sweep.py --snapshots checkpoints/interp-snapshots

Both are the same harness re-run, so this file holds only the loop, the
ordering, and the summary -- everything scientific lives in keyboard_probe.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

from typeshi.interp import digraph, empirical, layout

_spec = importlib.util.spec_from_file_location(
    "keyboard_probe", Path(__file__).parent / "keyboard_probe.py"
)
kp = importlib.util.module_from_spec(_spec)
sys.modules["keyboard_probe"] = kp
_spec.loader.exec_module(kp)


def sorted_snapshots(paths) -> list[Path]:
    """Numeric order by step. A lexical sort puts step-12000 before step-9000
    and the developmental trajectory reads backwards."""
    def step(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else -1
    return sorted((Path(p) for p in paths), key=step)


def summarize(runs: dict[str, dict]) -> list[dict]:
    """One flat row per run, for the metrics-versus-x plot and the results doc."""
    rows = []
    for name, block in runs.items():
        rows.append({
            "run": name,
            "distance_spearman": block["blind"]["metrics"]["distance_spearman"],
            "same_finger_auc": block["same_finger_auc"],
            "hand_accuracy": block["blind"]["metrics"]["hand_accuracy"],
            "neighbor_recall": block["blind"]["metrics"]["neighbor_recall"],
            "mean_position_error_u": block["blind"]["metrics"]["mean_position_error_u"],
            "permutation_p": block["blind"]["permutation_p"],
        })
    return rows


def sweep_wpm(model, tok, stats, wpms, seed: int = 0) -> dict[str, dict]:
    """Does the keyboard sharpen when the model is asked to type fast?

    Hand alternation should dominate more at speed, so a world model that gets
    crisper under load is a stronger claim than a static map.
    """
    runs = {}
    for wpm in wpms:
        latency = digraph.probe_matrix(model, tok, stats, wpm=wpm)
        runs[f"wpm={wpm:g}"] = kp.analyze(latency, stats.bigram_counts, seed)
    return runs


def sweep_checkpoints(paths, stats, wpm: float = 52.5, device: str = "cpu",
                      seed: int = 0) -> dict[str, dict]:
    """The keyboard condensing out of noise, one saved checkpoint at a time."""
    from transformers import AutoModelForCausalLM

    from typeshi.eval.load import load_checkpoint_tokenizer

    runs = {}
    for path in sorted_snapshots(paths):
        tok = load_checkpoint_tokenizer(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        latency = digraph.probe_matrix(model, tok, stats, wpm=wpm)
        runs[path.name] = kp.analyze(latency, stats.bigram_counts, seed)
        print(f"  {path.name}: rho="
              f"{runs[path.name]['blind']['metrics']['distance_spearman']:+.3f}")
    return runs


def write_figure(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for metric in ("distance_spearman", "same_finger_auc", "hand_accuracy"):
        ax.plot([r["run"] for r in rows], [r[metric] for r in rows],
                marker="o", label=metric)
    ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def write_small_multiples(runs: dict[str, dict], path: Path) -> None:
    """One recovered keyboard per checkpoint -- the trajectory as a picture.

    The metric plot says whether structure grew; this says what kind. A run
    whose rho is flat but whose keys go from a blob to two hand-shaped clusters
    is a different story, and only the panels show it.
    """
    import matplotlib.pyplot as plt

    truth = layout.truth_coords()
    names = list(runs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4),
                             squeeze=False)
    for ax, name in zip(axes[0], names):
        fitted = np.array(runs[name]["blind"]["coords"])
        ax.scatter(truth[:, 0], truth[:, 1], s=150, facecolors="none",
                   edgecolors="0.8")
        for key, (x, y) in zip(layout.KEYS, fitted):
            ax.text(x, y, "␣" if key == " " else key, ha="center", va="center",
                    fontsize=8)
        rho = runs[name]["blind"]["metrics"]["distance_spearman"]
        ax.set_title(f"{name}\nrho={rho:+.2f}", fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("checkpoints/interp-snapshots"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--limit", type=int, default=200_000)
    parser.add_argument("--wpms", type=float, nargs="+", default=[27.5, 82.5])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("keyboard_sweep.json"))
    parser.add_argument("--figure", type=Path, default=Path("keyboard_sweep.png"))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    from typeshi.eval.load import load_checkpoint_tokenizer

    stats = empirical.corpus_stats(args.data, limit=args.limit, seed=args.seed)
    snapshots = sorted_snapshots(args.snapshots.glob("step-*"))
    if not snapshots:
        raise SystemExit(f"no step-* directories under {args.snapshots}")

    print("checkpoint sweep...")
    by_step = sweep_checkpoints(snapshots, stats, device=args.device, seed=args.seed)

    print(f"wpm sweep on {snapshots[-1].name}...")
    latest = snapshots[-1]
    tok = load_checkpoint_tokenizer(latest)
    model = AutoModelForCausalLM.from_pretrained(latest).to(args.device)
    by_wpm = sweep_wpm(model, tok, stats, args.wpms, seed=args.seed)

    report = {
        "checkpoints": summarize(by_step),
        "wpm": summarize(by_wpm),
        "wpm_checkpoint": latest.name,
    }
    args.out.write_text(json.dumps(report, indent=2))
    write_figure(report["checkpoints"], args.figure)
    panels = args.figure.with_name(args.figure.stem + "_panels.png")
    write_small_multiples(by_step, panels)
    print(f"wrote {args.out}, {args.figure} and {panels}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_keyboard_sweep.py -v`
Expected: PASS, 2 tests

- [ ] **Step 5: Run both sweeps**

Run: `uv run python scripts/keyboard_sweep.py --snapshots checkpoints/interp-snapshots`
Expected: one row per preserved checkpoint plus the two WPM conditions, and a metrics-versus-step figure.

- [ ] **Step 6: Commit**

```bash
git add scripts/keyboard_sweep.py tests/test_keyboard_sweep.py
git commit -m "feat: WPM and checkpoint sweeps for the keyboard probe"
```

---

### Task 8: Results write-up

**Files:**
- Create: `docs/results-keyboard-reconstruction.md`

- [ ] **Step 1: Write the results doc**

Follow the structure of `docs/results-08b-shakedown.md`. It must contain, with actual measured numbers rather than descriptions of them:

1. The synthetic pipeline ceiling from Task 2, stated first — every model number is read against it.
2. The biomechanical table for model, corpus, and random-init side by side, with the class ordering called out explicitly.
3. Level-2 and level-3 metrics for blind and finger-aware, with the finger-aware column labelled ground-truth-assisted at the point of use.
4. The three controls: random-init, frequency-controlled, and permutation p-values.
5. The hold-bin sensitivity spread from Task 6 step 7 — all three values, not the best.
6. The WPM contrast and the checkpoint trajectory, with the figures inline.
7. An honest reading against the spec's §2 prediction: how much of the disagreement with physical QWERTY is the predicted same-finger inversion, and how much is unexplained.
8. What this does and does not say about the realism gates — it is evidence about the data that does not route through the discriminator, and nothing more.

- [ ] **Step 2: Commit**

```bash
git add docs/results-keyboard-reconstruction.md
git commit -m "docs: keyboard reconstruction results"
```
