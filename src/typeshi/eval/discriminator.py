"""Learned real-vs-generated classifier. Our pass condition is that this
model CANNOT tell the difference, so it must first be shown to have teeth."""

from __future__ import annotations

import numpy as np

from typeshi.events import Event
from typeshi.eval.distributional import timing_features

_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def serial_features(events: list[Event]) -> np.ndarray:
    """Order-sensitive features of one session's gap sequence.

    Marginal quantiles are blind to serial order by construction -- permuting
    a session's gaps leaves every one of them unchanged, which is why the
    first eval's serial-dependence gate scored 0.4975 (chance) against
    timing-shuffled real sessions. Each feature here has a known null under
    exchangeability, so shuffling drags it toward that null while real motor
    behaviour (drift, bursts, word-boundary slowdowns, hold/gap coupling)
    holds it away.

    Nine values, always finite, fixed length regardless of session size.
    """
    press = np.array([e.press_time for e in events], dtype=float)
    iki = np.diff(press) if len(press) > 1 else np.array([])
    x = np.log1p(np.maximum(iki, 0.0))
    n = x.size
    out: list[float] = []

    # Lag-k autocorrelation of log-gaps (null: 0).
    for k in (1, 2, 3):
        if n > k + 2 and x.std() > 1e-9:
            out.append(float(np.corrcoef(x[:-k], x[k:])[0, 1]))
        else:
            out.append(0.0)

    # Von Neumann ratio: mean squared successive difference over variance.
    # Exchangeable null: 2. Positive serial dependence pulls it below.
    if n > 3 and x.var() > 1e-12:
        out.append(float(np.mean(np.diff(x) ** 2) / x.var()))
    else:
        out.append(2.0)

    # Fast/slow Markov excess: P(fast -> fast) - P(fast). Null: 0.
    # Real typing clusters fast keystrokes into bursts.
    if n > 4:
        fast = x <= np.median(x)
        p_ff = float((fast[:-1] & fast[1:]).sum()) / max(int(fast[:-1].sum()), 1)
        out.append(p_ff - float(fast.mean()))
    else:
        out.append(0.0)

    # Local-to-global spread: mean std of 8-gap windows over the whole
    # session's std. Clustered slow/fast segments push it below 1; a
    # shuffle homogenises the windows back toward 1.
    if n >= 16 and x.std() > 1e-9:
        w = 8
        locals_ = [x[i:i + w].std() for i in range(0, n - w + 1, w)]
        out.append(float(np.mean(locals_) / x.std()))
    else:
        out.append(1.0)

    # Drift: |first-half mean - second-half mean| in session-std units.
    # Warmup and fatigue are real; a shuffle mixes the halves. Null: ~0.
    if n >= 8 and x.std() > 1e-9:
        h = n // 2
        out.append(float(abs(x[:h].mean() - x[h:].mean()) / x.std()))
    else:
        out.append(0.0)

    # Hold/gap coupling: corr(hold of key i, gap i -> i+1). Motor control
    # ties them (rollover, deliberate keystrokes); a shuffle decouples.
    if n > 3:
        holds = np.array(
            [
                np.nan if e.release_time is None
                else e.release_time - e.press_time
                for e in events[:-1]
            ],
            dtype=float,
        )
        m = ~np.isnan(holds)
        lh = np.log1p(np.maximum(holds[m], 0.0))
        if m.sum() > 3 and lh.std() > 1e-9 and x[m].std() > 1e-9:
            out.append(float(np.corrcoef(lh, x[m])[0, 1]))
        else:
            out.append(0.0)
    else:
        out.append(0.0)

    # Word-boundary slowdown: mean log-gap into a word-initial key minus the
    # mean elsewhere. Gap i runs from event i to i+1, so event i's char being
    # a space marks gap i as word-entering. Chars stay put under the timing
    # shuffle, so this collapses to ~0 there while real typing keeps it high.
    if n > 3:
        chars = [e.char for e in events[:-1]]
        into_word = np.array([c == " " for c in chars])
        if into_word.sum() >= 1 and (~into_word).sum() >= 3:
            out.append(float(x[into_word].mean() - x[~into_word].mean()))
        else:
            out.append(0.0)
    else:
        out.append(0.0)

    return np.nan_to_num(np.array(out, dtype=float))


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
    parts += list(serial_features(events))
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
    _, accuracy = train_discriminator(left, right, paired=False, seed=seed)
    return accuracy


def train_discriminator(
    real_sessions: list[list[Event]],
    fake_sessions: list[list[Event]],
    *,
    paired: bool,
    seed: int = 0,
    count_features: bool = True,
) -> tuple[object, float]:
    """Cross-validated real-vs-fake accuracy.

    `paired` has no default deliberately: forgetting it silently reintroduces
    the below-chance leak, so every caller must decide. It MUST be True whenever fake session i was generated for the
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
