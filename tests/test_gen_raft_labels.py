"""gen_raft_data must invert the prompt's knob bins EXACTLY.

The script's own defense is a byte-equality gate: rebuild the prompt from the
parsed labels and skip the row on mismatch. When <REV:> moved to geometric
bins, the rv/100 inversion stopped matching rev_bin, so every stored prompt
with REV>0 fails the gate and is silently skipped -- an entire behaviour
class vanishing from the preference pool with no error.
"""

import importlib.util
import sys
from pathlib import Path

from typeshi.dataset import build_prompt
from typeshi.labels import SessionLabels

_SPEC = importlib.util.spec_from_file_location(
    "gen_raft_data",
    Path(__file__).resolve().parent.parent / "scripts" / "gen_raft_data.py",
)
gen_raft_data = importlib.util.module_from_spec(_SPEC)
sys.modules["gen_raft_data"] = gen_raft_data
_SPEC.loader.exec_module(gen_raft_data)


def test_labels_from_prompt_round_trips_a_revising_row():
    labels = SessionLabels(
        wpm=42.0,
        corrected_error_rate=0.04,
        uncorrected_error_rate=0.01,
        revision_rate=0.012,  # a real writer's rate: geometric bin, not 1%
    )
    prompt = build_prompt("a b c", labels, "transcription")
    parsed = gen_raft_data.labels_from_prompt(prompt)
    assert parsed is not None
    assert build_prompt("a b c", parsed, "transcription") == prompt, \
        "the rebuilt prompt must be byte-identical or the row is skipped"


def test_labels_from_prompt_round_trips_the_zero_bin():
    labels = SessionLabels(42.0, 0.0, 0.0, 0.0)
    prompt = build_prompt("a b c", labels, "transcription")
    parsed = gen_raft_data.labels_from_prompt(prompt)
    assert parsed is not None
    assert build_prompt("a b c", parsed, "transcription") == prompt
