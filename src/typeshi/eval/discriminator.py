"""Learned real-vs-generated classifier. Our pass condition is that this
model CANNOT tell the difference, so it must first be shown to have teeth."""

from __future__ import annotations

import numpy as np

from typeshi.events import Event
from typeshi.eval.distributional import timing_features

_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def featurize(events: list[Event], count_features: bool = True) -> np.ndarray:
    """Summary vector of a session's timing.

    `count_features=False` drops the per-feature event counts. Counts make the
    classifier partly a LENGTH discriminator: a generator that types the target
    once, without a human's corrections, differs in event count before any
    timing is inspected. The eval reports both variants so a pass on the full
    vector can be checked against the timing-only one.
    """
    f = timing_features(events)
    parts: list[float] = []
    for key in ("iki", "hold", "pause", "burst"):
        x = f[key]
        if x.size == 0:
            parts += [0.0] * (len(_QUANTILES) + 2 + int(count_features))
            continue
        lx = np.log1p(x)
        parts += list(np.quantile(lx, _QUANTILES))
        parts += [float(lx.mean()), float(lx.std())]
        if count_features:
            parts.append(float(len(x)))
    # Autocorrelation of successive gaps: humans drift, naive samplers do not.
    iki = f["iki"]
    if iki.size > 2:
        li = np.log1p(iki)
        parts.append(float(np.corrcoef(li[:-1], li[1:])[0, 1]))
    else:
        parts.append(0.0)
    return np.nan_to_num(np.array(parts, dtype=float))


def heuristic_baseline(target_text: str, wpm: float, seed: int = 0) -> list[Event]:
    """Deliberately naive simulator: Gaussian jitter around a fixed mean gap.
    Stands in for off-the-shelf typing simulators as a discriminator control."""
    rng = np.random.default_rng(seed)
    mean_gap = 60_000 / (wpm * 5)
    events, t = [], 0
    for ch in target_text:
        gap = max(int(rng.normal(mean_gap, mean_gap * 0.15)), 1)
        hold = max(int(rng.normal(80, 12)), 1)
        events.append(Event.key(ch, t, t + hold))
        t += gap
    return events


def real_vs_real_control(
    real_sessions: list[list[Event]], seed: int = 0
) -> float:
    """Floor check: the discriminator scored on real-vs-real, which should
    land near chance. If it does not, the featurization is leaking something
    unrelated to realism and no real-vs-fake number from it can be trusted.

    The split is shuffled deliberately. Splitting a session list down the
    middle instead groups whole participants on one side, and the classifier
    then separates *typists* rather than real from fake: measured 0.86 on a
    sequential split of 200 Aalto sessions from 14 participants, against 0.42
    interleaved. For the same reason, real and generated sessions must be
    compared pairwise on the same targets, never as two independent pools.
    """
    import random as _random

    indices = list(range(len(real_sessions)))
    _random.Random(seed).shuffle(indices)
    half = len(indices) // 2
    left = [real_sessions[i] for i in indices[:half]]
    right = [real_sessions[i] for i in indices[half:]]
    if not left or not right:
        return float("nan")
    _, accuracy = train_discriminator(left, right, seed=seed)
    return accuracy


def train_discriminator(
    real_sessions: list[list[Event]],
    fake_sessions: list[list[Event]],
    seed: int = 0,
    paired: bool = False,
    count_features: bool = True,
) -> tuple[object, float]:
    """Cross-validated real-vs-fake accuracy.

    `paired=True` MUST be used whenever fake session i was generated for the
    same target as real session i. Plain stratified folds put one member of a
    pair in training and its opposite-labelled near-twin in validation, which
    biases accuracy BELOW chance -- measured 0.085 on exact-copy fakes, where
    the true answer is 0.50. A gate of the form "accuracy <= 0.55" rewards
    exactly that failure, so unpaired CV on paired data would pass a broken
    model. Grouped folds keep each pair on one side of every split.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import (
        StratifiedGroupKFold, StratifiedKFold, cross_val_score,
    )

    X = np.vstack(
        [featurize(s, count_features) for s in real_sessions]
        + [featurize(s, count_features) for s in fake_sessions]
    )
    y = np.array([1] * len(real_sessions) + [0] * len(fake_sessions))
    clf = GradientBoostingClassifier(random_state=seed)
    # Explicit seeded splitters so the reported accuracy is reproducible.
    if paired:
        if len(real_sessions) != len(fake_sessions):
            raise ValueError("paired CV needs equally many real and fake sessions")
        groups = np.tile(np.arange(len(real_sessions)), 2)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        acc = float(
            cross_val_score(clf, X, y, cv=cv, groups=groups, scoring="accuracy").mean()
        )
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())
    clf.fit(X, y)
    return clf, acc


def shuffle_timing(events: list[Event], seed: int = 0) -> list[Event]:
    """Hard negative: a real session with its inter-press gaps permuted.

    Chars, holds, and the gap MULTISET are untouched -- only serial order
    changes. A discriminator with real teeth must catch this (humans have
    strong serial dependence); one that only reads marginal distributions
    cannot, so this is reported as a second, harder teeth check.
    """
    import dataclasses
    import random as _random

    if len(events) < 3:
        return list(events)
    press = [e.press_time for e in events]
    gaps = [b - a for a, b in zip(press, press[1:])]
    _random.Random(seed).shuffle(gaps)
    out, t = [], 0
    for i, e in enumerate(events):
        if i > 0:
            t += gaps[i - 1]
        hold = None if e.release_time is None else e.release_time - e.press_time
        out.append(dataclasses.replace(
            e, press_time=t, release_time=None if hold is None else t + hold,
        ))
    return out
