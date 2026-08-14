"""Mirrors run_eval's generation loop; reports similarity + failure text."""
import json
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from typeshi.adapters import aalto
from typeshi.buffer import replay
from typeshi.events import EventType
from typeshi.generate import generate
from typeshi.labels import _levenshtein, compute_labels

ck = "checkpoints/motor"
tok = AutoTokenizer.from_pretrained(ck)
model = AutoPeftModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, device_map="auto")
model.eval()

test_writers = set(json.load(open(f"{ck}/split.json"))["test_writers"])

cases = []
for writer, target, events in aalto.iter_sessions("data/heldout_writers"):
    if f"aalto:{writer}" not in test_writers:
        continue
    cases.append((target, compute_labels(events, target)))
    if len(cases) >= 15:
        break
print(f"{len(cases)} held-out cases", flush=True)

for temp in (1.0, 0.7, 0.5, 0.3):
    sims, shown = [], 0
    bad_types = 0
    for i, (target, labels) in enumerate(cases):
        budget = min(512, 4 * len(target) + 64)
        try:
            events = generate(model, tok, target, labels,
                              temperature=temp, max_new_tokens=budget, seed=i)
        except ValueError as e:
            sims.append(-1.0)  # malformed
            continue
        if any(e.type not in (EventType.KEY, EventType.BACKSPACE) for e in events):
            bad_types += 1
            sims.append(-0.5)
            continue
        produced = replay(events)
        sim = 1 - _levenshtein(produced, target) / max(len(target), 1)
        sims.append(sim)
        if sim < 0.8 and shown < 2:
            shown += 1
            print(f"  [t={temp} sim={sim:.2f}]", flush=True)
            print(f"    target:   {target[:70]!r}")
            print(f"    produced: {produced[:70]!r}")
    valid = sum(s >= 0.8 for s in sims)
    srt = sorted(sims)
    print(f"temp {temp}: valid {valid}/{len(cases)}  malformed {sum(s==-1.0 for s in sims)}  "
          f"badtypes {bad_types}  median {srt[len(srt)//2]:.3f}", flush=True)
print("PROBE DONE", flush=True)
