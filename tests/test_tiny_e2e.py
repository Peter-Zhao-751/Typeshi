# tests/test_tiny_e2e.py
"""Fixture-scale proof of the whole tiny loop: train -> save -> reload from
disk the way run_eval does -> constrained generation -> EOS before budget ->
deserialize. Overfits ONE short session on purpose: a memorized model must
emit EOS at the stream's end, so a hang here means the termination path is
broken (the exact failure the EOS-in-gap fix exists to prevent)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

from typeshi.adapters import aalto
from typeshi.dataset import build_examples
from typeshi.labels import compute_labels

FIXTURE = Path(__file__).parent / "fixtures" / "aalto_sample.txt"


@pytest.mark.slow
def test_train_reload_generate_terminate(tmp_path, monkeypatch):
    sessions = list(aalto.iter_sessions(FIXTURE))
    assert sessions, "fixture parsed to zero sessions"
    _, target, events = sessions[0]
    labels = compute_labels(events, target)
    examples = build_examples(target, events, labels, "transcription")

    train_file = tmp_path / "train.jsonl"
    with train_file.open("w") as f:
        for _ in range(800):  # ~100 optimizer steps at batch 8
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    out = tmp_path / "ckpt"
    from typeshi import train_tiny

    monkeypatch.setattr(sys, "argv", [
        "train_tiny", "--data", str(train_file), "--out", str(out),
        "--config", "smoke", "--epochs", "1", "--batch", "8", "--accum", "1",
        "--lr", "1e-2",
    ])
    train_tiny.main()

    # Reload EXACTLY as run_eval.py does: from disk, via the auto classes.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    model.eval()

    from typeshi.generate import generate

    budget = 4 * len(target) + 64
    gen = generate(model, tok, target, labels, mode="transcription",
                   temperature=0.7, max_new_tokens=budget, seed=0)
    assert gen, "constrained generation produced no events"
    # Termination proof: generation stops only on EOS or budget exhaustion.
    # A full-budget stream is ~budget/2 events; well under that means EOS.
    assert 2 * len(gen) + 2 < budget, "burned the whole budget -- EOS never came"
