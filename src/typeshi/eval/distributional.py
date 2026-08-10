"""Distributional fidelity metrics between real and generated timing."""

from __future__ import annotations

import numpy as np

from typeshi.events import Event, EventType

PAUSE_THRESHOLD_MS = 1000


def timing_features(events: list[Event]) -> dict[str, np.ndarray]:
    press = np.array([e.press_time for e in events], dtype=float)
    iki = np.diff(press) if len(press) > 1 else np.array([])
    hold = np.array(
        [e.release_time - e.press_time for e in events if e.release_time is not None],
        dtype=float,
    )
    pause = iki[iki > PAUSE_THRESHOLD_MS] if iki.size else np.array([])

    bursts, run = [], 1
    for gap in iki:
        if gap > PAUSE_THRESHOLD_MS:
            bursts.append(run)
            run = 1
        else:
            run += 1
    if len(events):  # an empty session has no burst, not a burst of one
        bursts.append(run)

    return {"iki": iki, "hold": hold, "pause": pause,
            "burst": np.array(bursts, dtype=float)}


def pooled_features(sessions: list[list[Event]]) -> dict[str, np.ndarray]:
    """Timing features pooled across many sessions.

    Empty sessions contribute nothing; an all-empty collection therefore
    produces empty pools and NaN metrics downstream (serialized as null in
    the report). The eval runner cannot feed empties here -- its validity
    gate rejects them upstream -- so no count of dropped sessions is kept.

    Sessions must not be concatenated into one event list before measuring:
    every session is rebased to zero, so a naive concatenation invents a large
    negative inter-key interval at each boundary, which poisons the IKI
    distribution and makes the log-domain Frechet distance NaN. Features are
    computed per session and only then pooled.
    """
    keys = ("iki", "hold", "pause", "burst")
    parts: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for session in sessions:
        if not session:
            continue
        feats = timing_features(session)
        for k in keys:
            if feats[k].size:
                parts[k].append(feats[k])
    return {
        k: np.concatenate(v) if v else np.array([]) for k, v in parts.items()
    }


MIN_SAMPLES = 10


def _validate_timings(name: str, x: np.ndarray) -> np.ndarray:
    """Rejects impossible timings, tolerates the boundary case.

    Negative or non-finite values are caller bugs (e.g. concatenating rebased
    sessions) and raise. ZERO is legitimate: the Aalto adapter clamps logging
    jitter with release = max(release, press), so a zero hold occurs in real
    data (~1 in 38k events) -- rejecting it crashed the whole eval. Zeros are
    lifted to the histogram floor instead of being silently excluded by the
    log-spaced edges.
    """
    if x.size and (not np.isfinite(x).all() or (x < 0).any()):
        raise ValueError(f"{name}_samples must be finite and non-negative")
    return np.maximum(x, 1e-3) if x.size else x


def kl_divergence(p_samples, q_samples, bins: int = 64) -> float:
    """Symmetrized KL over a shared log-spaced histogram."""
    p_samples = np.asarray(p_samples, dtype=float)
    q_samples = np.asarray(q_samples, dtype=float)
    p_samples = _validate_timings("p", p_samples)
    q_samples = _validate_timings("q", q_samples)
    # Below ~10 samples the +1e-9 smoothing dominates the estimate (a single
    # 5 vs a single 7 scores ~20 bits of "divergence"), so refuse to guess.
    if p_samples.size < MIN_SAMPLES or q_samples.size < MIN_SAMPLES:
        return float("nan")

    lo = max(min(p_samples.min(), q_samples.min()), 1e-3)
    hi = max(p_samples.max(), q_samples.max())
    edges = np.geomspace(lo, hi + 1, bins + 1)

    p, _ = np.histogram(p_samples, bins=edges)
    q, _ = np.histogram(q_samples, bins=edges)
    p = (p + 1e-9) / (p.sum() + 1e-9 * bins)
    q = (q + 1e-9) / (q.sum() + 1e-9 * bins)
    return float(0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))))


def frechet_distance(p_samples, q_samples) -> float:
    """1-D Frechet distance: (mu1-mu2)^2 + (s1-s2)^2, on log-transformed times.

    Shares kl_divergence's guards: a singleton previously scored a finite,
    misleadingly "perfect" 0.0 against itself, and negative/NaN input sailed
    through log1p as warnings rather than errors.
    """
    p = _validate_timings("p", np.asarray(p_samples, dtype=float))
    q = _validate_timings("q", np.asarray(q_samples, dtype=float))
    if p.size < MIN_SAMPLES or q.size < MIN_SAMPLES:
        return float("nan")
    p, q = np.log1p(p), np.log1p(q)
    return float((p.mean() - q.mean()) ** 2 + (p.std() - q.std()) ** 2)


def _scores(real: dict[str, np.ndarray], fake: dict[str, np.ndarray]) -> dict:
    return {
        key: {
            "kl": kl_divergence(real[key], fake[key]),
            "frechet": frechet_distance(real[key], fake[key]),
        }
        for key in real
    }


def compare(real_events: list[Event], fake_events: list[Event]) -> dict:
    """Compares two single sessions."""
    return _scores(timing_features(real_events), timing_features(fake_events))


def compare_sessions(
    real_sessions: list[list[Event]], fake_sessions: list[list[Event]]
) -> dict:
    """Compares two *collections* of sessions. Use this for eval reports."""
    return _scores(pooled_features(real_sessions), pooled_features(fake_sessions))
