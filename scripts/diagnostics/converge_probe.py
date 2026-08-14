"""Convergence decoder vs the same 5 held-out KLiCKe prompts (2/5 free)."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from typeshi.adapters import klicke
from typeshi.buffer import replay
from typeshi.eval.load import load_checkpoint_model, load_checkpoint_tokenizer
from typeshi.generate import generate
from typeshi.labels import compute_labels
from typeshi.train_motor import _detect_backend

CK = Path("checkpoints/motor-phase2")
tok = load_checkpoint_tokenizer(CK)
model = load_checkpoint_model(CK, _detect_backend())
model.eval()
test_writers = set(json.load(open(CK / "split.json"))["test_writers"])

shown = 0
exact = 0
for path in sorted(Path("data/raw/klicke").rglob("*.csv")):
    if klicke.gold_text_path(path) is None:
        continue
    for writer, final_text, events in klicke.iter_sessions(path):
        if f"klicke:{writer}" not in test_writers:
            continue
        target = final_text[:220]  # bounded prompt for probe speed
        labels = compute_labels(events, final_text)
        try:
            gen = generate(model, tok, target, labels, mode="composition",
                           temperature=1.0, max_new_tokens=4 * len(target) + 256,
                           seed=shown, constrained=True)
        except ValueError as e:
            print(f"[{shown}] MALFORMED: {e}", flush=True)
            shown += 1
            break
        produced = replay(gen)
        ok = produced == target
        exact += ok
        mix = Counter(e.type.name for e in gen)
        n = max(sum(mix.values()), 1)
        ikis = np.diff([e.press_time for e in gen])
        print(f"[{shown}] exact={ok}  events={len(gen)}  "
              f"bksp={mix.get('BACKSPACE',0)/n:.3f}  "
              f"pause>1s={(ikis>1000).mean():.4f}", flush=True)
        if not ok:
            print(f"    produced tail: {produced[-80:]!r}")
        shown += 1
        break
    if shown >= 50:
        break
print(f"CONVERGED {exact}/{shown}", flush=True)
print("CONVERGE PROBE DONE", flush=True)
