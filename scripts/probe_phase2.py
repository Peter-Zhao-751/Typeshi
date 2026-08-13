"""Phase-2 checkpoint sanity probes.

A) Transcription regression: constrained generation on held-out Aalto
   sentences -- validity must not have been destroyed by the mixed curriculum.
B) Composition behaviour: unconstrained <MODE:C> generation on held-out
   KLiCKe essays -- event-type mix, pause structure, parseability.
"""
import json
from collections import Counter

import numpy as np

from typeshi.adapters import aalto, klicke
from typeshi.buffer import replay
from typeshi.events import EventType
from typeshi.eval.load import load_checkpoint_model, load_checkpoint_tokenizer
from typeshi.generate import generate
from typeshi.labels import _levenshtein, compute_labels
from typeshi.train_motor import _detect_backend

from pathlib import Path
CK = Path("checkpoints/motor-phase2")
backend = _detect_backend()
tok = load_checkpoint_tokenizer(CK)
model = load_checkpoint_model(CK, backend)
model.eval()

split = json.load(open(f"{CK}/split.json"))
test_writers = set(split["test_writers"])

# ---- A) transcription regression --------------------------------------
n_ok, n = 0, 0
for writer, target, events in aalto.iter_sessions("data/heldout_writers"):
    labels = compute_labels(events, target)
    try:
        gen = generate(model, tok, target, labels, mode="transcription",
                       temperature=1.0, max_new_tokens=4 * len(target) + 64, seed=n)
        produced = replay(gen)
        sim = 1 - _levenshtein(produced, target) / max(len(target), 1)
        n_ok += sim >= 0.8
    except ValueError:
        pass
    n += 1
    if n >= 20:
        break
print(f"[A] transcription validity: {n_ok}/{n}", flush=True)

# ---- B) composition behaviour -----------------------------------------
def event_mix(events):
    c = Counter(e.type.name for e in events)
    total = max(sum(c.values()), 1)
    return {k: round(v / total, 3) for k, v in sorted(c.items())}

def pause_frac(events):
    ikis = np.diff([e.press_time for e in events])
    return round(float((ikis > 1000).mean()), 4) if ikis.size else 0.0

shown = 0
mal = 0
for path in sorted(__import__("pathlib").Path("data/raw/klicke").rglob("*.csv")):
    if klicke.gold_text_path(path) is None:
        continue
    for writer, final_text, events in klicke.iter_sessions(path):
        if f"klicke:{writer}" not in test_writers:
            continue
        labels = compute_labels(events, final_text)
        try:
            gen = generate(model, tok, final_text, labels, mode="composition",
                           temperature=1.0, max_new_tokens=1500,
                           seed=shown, constrained=False)
        except ValueError:
            mal += 1
            continue
        from typeshi.buffer import ReplayError
        try:
            produced = replay(gen)
        except ReplayError as e:
            produced = f"<REPLAY ERROR: {e}>"
        print(f"[B] session {shown}: {len(gen)} events", flush=True)
        print(f"    real mix: {event_mix(events)}  pause>1s: {pause_frac(events)}")
        print(f"    gen  mix: {event_mix(gen)}  pause>1s: {pause_frac(gen)}")
        print(f"    gen text head: {produced[:100]!r}")
        print(f"    target head:   {final_text[:100]!r}")
        shown += 1
        break
    if shown >= 5:
        break
print(f"[B] malformed (unconstrained): {mal}", flush=True)
print("PROBES DONE", flush=True)
