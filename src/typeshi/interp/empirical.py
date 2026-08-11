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
