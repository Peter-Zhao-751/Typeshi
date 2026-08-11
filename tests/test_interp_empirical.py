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
