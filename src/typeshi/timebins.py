"""Log-spaced quantization of inter-event times.

Typing spans three orders of magnitude (2 ms rollover to 60 s thinking pause),
so bins are geometric: fine where keystrokes are fast, coarse where pauses are long.
"""

from __future__ import annotations

import functools

import numpy as np

from typeshi import config


@functools.lru_cache(maxsize=1)
def bin_edges() -> np.ndarray:
    return np.geomspace(config.MIN_MS, config.MAX_MS, config.TIME_BINS + 1)


def to_bin(dt_ms: int) -> int:
    edges = bin_edges()
    dt = min(max(float(dt_ms), config.MIN_MS), config.MAX_MS)
    # searchsorted returns the insertion index; subtract 1 for the containing bin.
    k = int(np.searchsorted(edges, dt, side="right")) - 1
    return min(max(k, 0), config.TIME_BINS - 1)


def from_bin(k: int) -> int | float:
    if not 0 <= k < config.TIME_BINS:
        raise ValueError(f"bin index {k} out of range")
    edges = bin_edges()
    # Return the geometric mean (bin center) as a float to guarantee round-trip
    # idempotence: to_bin(from_bin(k)) == k for all k. Integer rounding breaks
    # this for low-valued bins where resolution is coarse.
    return float(np.sqrt(edges[k] * edges[k + 1]))
