"""Per-sample realism readout.

What can honestly be said about ONE session is narrower than what the eval
report says about a corpus, and the gap is not small. Two rules follow:

  * No per-sample KL or Frechet. Both guard on MIN_SAMPLES=10 per side, and
    `pause`/`burst` are empty or singleton for a single session, so they
    return NaN. Worse, where they do compute they are dominated by histogram
    sparsity: a REAL held-out human scores iki KL 3.59 against the pooled
    real reference where the report's pooled figure is 0.014. Surfacing that
    would mark every genuine human as inhuman.

  * No P(real) dial. The honest one is a persisted discriminator's
    predict_proba, but no such artifact exists -- train_discriminator returns
    a fitted classifier and every call site discards it -- and fitting one
    needs 200 held-out pairs, which is hours of decoding at this model's
    speed. Until that exists, claiming a calibrated probability would be
    inventing a number.

What IS honest per-sample: the nine order-sensitive features, because each
has a known null under exchangeability. A gauge reading "null | this sample |
real-human band" needs no reference pool for its null and no calibration for
its meaning, and it is exactly where the residual discriminator signal lives
(hold/gap coupling is the top GBM feature at 0.092).
"""

from __future__ import annotations

import numpy as np

from typeshi.buffer import replay
from typeshi.config import REPLAY_SIM_MIN
from typeshi.events import Event, EventType
from typeshi.eval.discriminator import (
    heuristic_baseline,
    serial_features,
    shuffle_timing,
)
from typeshi.labels import _levenshtein
from typeshi.serialize import codec_roundtrip

# Order matches serial_features()'s output exactly. `null` is the value the
# feature takes when gaps are exchangeable -- i.e. what a timing shuffle
# drags it toward and what a memoryless simulator sits at.
SERIAL_FEATURES: list[tuple[str, str, float]] = [
    ("autocorr_lag1", "gap autocorrelation, lag 1", 0.0),
    ("autocorr_lag2", "gap autocorrelation, lag 2", 0.0),
    ("autocorr_lag3", "gap autocorrelation, lag 3", 0.0),
    ("von_neumann", "von Neumann ratio", 2.0),
    ("burst_markov", "burst clustering (Markov excess)", 0.0),
    ("local_spread", "local/global spread", 1.0),
    ("drift", "drift across the session", 0.0),
    ("hold_gap_coupling", "hold/gap coupling", 0.0),
    ("word_boundary", "word-boundary slowdown", 0.0),
]


def validity(events: list[Event], target_text: str, mode: str) -> dict:
    """The per-sample form of the Tier-1 validity gate.

    Transcription is the gated one: KEY/BACKSPACE only, replayed text within
    REPLAY_SIM_MIN of the target. Composition legitimately emits cursor ops,
    so the event-type half is reported without failing the sample.
    """
    if not events:
        return {"ok": False, "similarity": 0.0, "reason": "no events"}
    produced = replay_or_empty(events)
    similarity = 1 - _levenshtein(produced, target_text) / max(len(target_text), 1)
    off_type = sorted(
        {e.type.value for e in events
         if e.type not in (EventType.KEY, EventType.BACKSPACE)}
    )
    reasons = []
    if similarity < REPLAY_SIM_MIN:
        reasons.append(f"replay similarity {similarity:.3f} < {REPLAY_SIM_MIN}")
    if off_type and mode == "transcription":
        reasons.append(f"non-transcription events: {', '.join(off_type)}")
    return {
        "ok": not reasons,
        "similarity": round(float(similarity), 3),
        "threshold": REPLAY_SIM_MIN,
        "off_type_events": off_type,
        "reason": "; ".join(reasons) or "counts as valid",
    }


def replay_or_empty(events: list[Event]) -> str:
    from typeshi.buffer import ReplayError

    try:
        return replay(events)
    except ReplayError:
        from typeshi.portal.rows import replay_safe

        return replay_safe(events)[0]


def serial_readout(events: list[Event],
                   band: dict[str, list[float]] | None = None) -> list[dict]:
    values = serial_features(events)
    out = []
    for (key, label, null), value in zip(SERIAL_FEATURES, values):
        # float() before comparing, not just before storing: numpy scalars
        # compare to numpy bools, which json.dumps cannot serialize and which
        # fail as an opaque "Object of type bool is not JSON serializable".
        v = round(float(value), 4)
        row = {"key": key, "label": label, "null": null, "value": v}
        if band and key in band:
            p10, p50, p90 = band[key]
            row["p10"], row["median"], row["p90"] = p10, p50, p90
            row["in_band"] = bool(p10 <= v <= p90)
        out.append(row)
    return out


def real_band(sessions: list[list[Event]]) -> dict[str, list[float]]:
    """p10 / median / p90 of each serial feature over real human sessions.

    Every session is pushed through the codec first. Real corpus timings are
    raw milliseconds while generated ones are born on the 128-bin geometric
    grid, and comparing the two directly is how the harness scored 0.915 on
    quantization alone -- the band has to sit on the same grid as the sample
    being judged against it.
    """
    if not sessions:
        return {}
    rows = np.array([serial_features(codec_roundtrip(s)) for s in sessions])
    band = {}
    for i, (key, _label, _null) in enumerate(SERIAL_FEATURES):
        col = rows[:, i]
        band[key] = [round(float(np.percentile(col, p)), 4) for p in (10, 50, 90)]
    return band


def controls(events: list[Event], target_text: str, wpm: float,
             seed: int = 0) -> dict:
    """Two free reference points that make the gauges self-explaining.

    The naive simulator is what the eval catches 100% of the time; the same
    sample with its gaps shuffled is what it catches 80% of the time. Seeing
    all three on one axis says what "human-like serial structure" means
    without anyone reading the eval code.
    """
    out: dict[str, list[dict]] = {}
    try:
        out["heuristic"] = serial_readout(heuristic_baseline(target_text, wpm, seed))
    except Exception:  # noqa: BLE001 - a control is a nicety, never the answer
        out["heuristic"] = []
    if len(events) > 4:
        try:
            out["shuffled"] = serial_readout(shuffle_timing(events, seed))
        except Exception:  # noqa: BLE001
            out["shuffled"] = []
    return out
