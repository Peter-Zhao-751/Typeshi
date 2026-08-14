"""Knob-fidelity measurement (design spec section 8.3): sweep the requested
condition knobs, measure what the generated sessions actually realize.

Two numbers per knob because each catches what the other misses: Pearson r
says whether turning the dial moves behaviour in the right direction, MAE
says whether the scale is calibrated. A model realizing exactly half the
requested WPM scores r = 1.0 with a huge MAE; one pinned at the population
mean scores a modest MAE with no correlation at all.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from typeshi.events import Event
from typeshi.labels import SessionLabels, compute_labels

# Every SessionLabels field is a prompt knob (labels.py: "these become the
# prompt knobs"), so the report keys derive from the dataclass instead of a
# second hand-kept list that could drift.
KNOBS = tuple(f.name for f in fields(SessionLabels))


def realized_labels(events: list[Event], target_text: str) -> SessionLabels:
    """Realized knob values of one generated session.

    Delegates to compute_labels -- the exact measurement that produced the
    training labels -- so a requested-vs-realized gap can only mean the model
    missed the condition, never that eval and training defined WPM or an
    error rate differently.
    """
    return compute_labels(events, target_text)


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, nan where undefined.

    Sweeps routinely hold a knob constant (fixed EUNC while WPM varies), so
    zero variance is the normal case here, not an error. np.corrcoef would
    warn and divide float noise around the constant mean -- which can land on
    a spurious +/-1 -- so constancy is detected exactly via ptp == 0 and
    reported as nan, which the eval runner's jsonable() serializes as null.
    """
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def knob_fidelity(
    requested: list[SessionLabels], realized: list[SessionLabels]
) -> dict:
    """Per-knob fidelity over paired (requested, realized) sessions.

    Returns {knob: {"r": ..., "mae": ...}} for every SessionLabels field.
    Values are floats; r is nan when fewer than two pairs or a constant knob
    leaves correlation undefined, and both are nan over zero pairs (the eval
    runner never sends zero, but a crash there would take the whole report
    down with it). Mismatched lengths are a caller bug and raise.
    """
    if len(requested) != len(realized):
        raise ValueError(
            "requested and realized must pair up: "
            f"{len(requested)} vs {len(realized)}"
        )
    report: dict[str, dict[str, float]] = {}
    for knob in KNOBS:
        x = np.array([getattr(s, knob) for s in requested], dtype=float)
        y = np.array([getattr(s, knob) for s in realized], dtype=float)
        mae = float(np.mean(np.abs(x - y))) if x.size else float("nan")
        report[knob] = {"r": _pearson(x, y), "mae": mae}
    return report


def sweep_grid(
    base_labels: SessionLabels,
    wpm_values: list[float],
    ecor_values: list[float],
) -> list[SessionLabels]:
    """Requested conditions for an eval sweep: the full wpm x ecor product,
    remaining knobs inherited from base_labels unchanged.

    Pure data on purpose: generation and measurement live with the model
    backends, so one grid drives any of them and stays testable offline.
    WPM and corrected-error-rate are the two knobs the design sweeps (spec
    section 8.3); the others ride along fixed at their base values.
    """
    return [
        replace(base_labels, wpm=float(w), corrected_error_rate=float(e))
        for w in wpm_values
        for e in ecor_values
    ]
