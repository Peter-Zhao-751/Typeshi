"""Tier-2 composition-signature metrics (design spec section 8, item 4).

Tier-1 (distributional.py) scores marginal timing pools -- motor realism.
These signatures score *composition* behaviour: where pauses fall relative
to linguistic structure, how much gets produced between them, and what the
revision traffic looks like, compared against KLiCKe held-out writers.
"""

from __future__ import annotations

import numpy as np

from typeshi.buffer import TextBuffer
from typeshi.events import Event, EventType
from typeshi.eval.distributional import frechet_distance, kl_divergence

# Writing-research P-burst convention (Wengelin): production between pauses
# of two seconds or more. Deliberately not distributional.PAUSE_THRESHOLD_MS:
# composition pauses live on a slower scale than motor hesitations.
P_BURST_THRESHOLD_MS = 2000

PAUSE_CLASSES = ("clause_boundary", "word_boundary", "within_word")

_CLAUSE_CHARS = frozenset(".,;!?")


def pause_at_boundaries(events: list[Event]) -> dict[str, np.ndarray]:
    """Inter-key intervals split by the linguistic position of the key.

    Position is read from the simulated buffer at the moment the key goes
    down, never from the typed order: after a backspace the last *typed*
    char and the char before the caret disagree, and only the buffer says
    where the writer actually is. Text before the caret ending in clause
    punctuation plus space is a clause/sentence boundary; ending in any
    other space is a word boundary; everything else (including offset 0 and
    after newlines) is within-word -- the spec names exactly three classes,
    so the catch-all stays a catch-all.

    Intervals are press-to-press from the previous event of any type, the
    same diff timing_features takes. Intervals that end in BKSP/CURSOR/
    SELDEL are revision traffic and belong to revision_ops, not here.
    """
    out: dict[str, list[float]] = {c: [] for c in PAUSE_CLASSES}
    buf = TextBuffer()
    prev_press: int | float | None = None
    for e in events:
        if e.type is EventType.KEY and prev_press is not None:
            before = buf.text[: buf.cursor]
            if len(before) >= 2 and before[-1] == " " and before[-2] in _CLAUSE_CHARS:
                cls = "clause_boundary"
            elif before.endswith(" "):
                cls = "word_boundary"
            else:
                cls = "within_word"
            out[cls].append(float(e.press_time - prev_press))
        buf.apply(e)
        prev_press = e.press_time
    return {c: np.array(v, dtype=float) for c, v in out.items()}


def p_bursts(
    events: list[Event], threshold_ms: float = P_BURST_THRESHOLD_MS
) -> np.ndarray:
    """Keystroke counts between pauses longer than `threshold_ms`.

    Same run-length construction as timing_features' Tier-1 burst -- every
    event is one keystroke (a CURSOR event is a collapsed click or arrow
    run), a strict `>` splits, an empty session has no burst rather than a
    burst of zero -- so the two features differ only in threshold.
    """
    press = np.array([e.press_time for e in events], dtype=float)
    if press.size == 0:
        return np.array([])
    bursts, run = [], 1
    for gap in np.diff(press):
        if gap > threshold_ms:
            bursts.append(run)
            run = 1
        else:
            run += 1
    bursts.append(run)
    return np.array(bursts, dtype=float)


def revision_ops(events: list[Event]) -> dict:
    """Revision-op statistics for one session.

    `bksp_frac` is backspaces over ALL events (an empty session scores 0.0);
    `revision_episode_lengths` are maximal consecutive-BKSP run lengths --
    one three-key deletion is one episode of 3, not three of 1, which is
    what separates burst-erasers from single-typo correctors.
    """
    n_bksp = cursor_count = seldel_count = run = 0
    episodes: list[int] = []
    for e in events:
        if e.type is EventType.BACKSPACE:
            n_bksp += 1
            run += 1
            continue
        if run:
            episodes.append(run)
            run = 0
        if e.type is EventType.CURSOR:
            cursor_count += 1
        elif e.type is EventType.SELDEL:
            seldel_count += 1
    if run:
        episodes.append(run)
    return {
        "bksp_frac": n_bksp / len(events) if events else 0.0,
        "cursor_count": cursor_count,
        "seldel_count": seldel_count,
        "revision_episode_lengths": np.array(episodes, dtype=float),
    }


def pooled_signatures(sessions: list[list[Event]]) -> dict[str, np.ndarray]:
    """Signature features pooled across many sessions.

    Computed per session and only then pooled, never on a concatenated
    event list: every session is rebased to zero, so concatenation invents
    a large negative interval at each boundary (see
    distributional.pooled_features). Empty sessions contribute nothing.
    The revision scalars contribute one sample per session, so their
    metrics go NaN below MIN_SAMPLES sessions -- the metrics refuse to
    guess rather than score a handful of scalars.
    """
    pause_parts: dict[str, list[np.ndarray]] = {c: [] for c in PAUSE_CLASSES}
    burst_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    scalars: dict[str, list[float]] = {
        "bksp_frac": [], "cursor_count": [], "seldel_count": []
    }
    for session in sessions:
        if not session:
            continue
        pauses = pause_at_boundaries(session)
        for c in PAUSE_CLASSES:
            if pauses[c].size:
                pause_parts[c].append(pauses[c])
        burst_parts.append(p_bursts(session))
        ops = revision_ops(session)
        if ops["revision_episode_lengths"].size:
            episode_parts.append(ops["revision_episode_lengths"])
        for k in scalars:
            scalars[k].append(float(ops[k]))

    pooled = {
        f"pause_{c}": np.concatenate(v) if v else np.array([])
        for c, v in pause_parts.items()
    }
    pooled["p_burst"] = np.concatenate(burst_parts) if burst_parts else np.array([])
    pooled["revision_episode_lengths"] = (
        np.concatenate(episode_parts) if episode_parts else np.array([])
    )
    for k, v in scalars.items():
        pooled[k] = np.array(v, dtype=float)
    return pooled


def compare_signatures(
    real_sessions: list[list[Event]], fake_sessions: list[list[Event]]
) -> dict:
    """Scores every signature class between two session collections.

    Returns {class: {"kl": float, "frechet": float}} -- plain floats, NaN
    where a pool is too small to score (run_eval's jsonable() writes NaN
    as null in the report).
    """
    real = pooled_signatures(real_sessions)
    fake = pooled_signatures(fake_sessions)
    return {
        key: {
            "kl": kl_divergence(real[key], fake[key]),
            "frechet": frechet_distance(real[key], fake[key]),
        }
        for key in real
    }
