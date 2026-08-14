"""Where does the model's pause distribution diverge from real typists'?

Generates 50 held-out sessions, then compares inter-key-interval histograms
(log-spaced) between real and generated, focusing on the pause tail.
"""
import json

import numpy as np
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from typeshi.adapters import aalto
from typeshi.generate import generate
from typeshi.labels import compute_labels

ck = "checkpoints/motor"
tok = AutoTokenizer.from_pretrained(ck)
model = AutoPeftModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, device_map="auto")
model.eval()

test_writers = set(json.load(open(f"{ck}/split.json"))["test_writers"])


def ikis(events):
    times = [e.press_time for e in events]
    return np.diff(times)


real_ikis, gen_ikis = [], []
n = 0
for writer, target, events in aalto.iter_sessions("data/heldout_writers"):
    if f"aalto:{writer}" not in test_writers:
        continue
    labels = compute_labels(events, target)
    try:
        gen = generate(model, tok, target, labels, temperature=1.0,
                       max_new_tokens=min(512, 4 * len(target) + 64), seed=n)
    except ValueError:
        continue
    real_ikis.append(ikis(events))
    gen_ikis.append(ikis(gen))
    n += 1
    if n >= 50:
        break

real = np.concatenate(real_ikis).astype(float)
gen = np.concatenate(gen_ikis).astype(float)
print(f"{n} sessions | real IKIs {len(real)}, gen IKIs {len(gen)}")

edges = [0, 100, 200, 300, 500, 750, 1000, 2000, 5000, 10000, 30000, 1e9]
names = ["<100ms", "100-200", "200-300", "300-500", "500-750", "750-1k",
         "1-2s", "2-5s", "5-10s", "10-30s", ">30s"]
rh, _ = np.histogram(real, bins=edges)
gh, _ = np.histogram(gen, bins=edges)
print(f"{'bin':>10} {'real%':>8} {'gen%':>8}")
for name, r, g in zip(names, rh, gh):
    print(f"{name:>10} {100*r/len(real):>7.2f}% {100*g/len(gen):>7.2f}%")

for q in (50, 90, 99, 99.9):
    print(f"p{q}: real {np.percentile(real, q):>8.0f} ms   "
          f"gen {np.percentile(gen, q):>8.0f} ms")
print(f"max: real {real.max():.0f} ms   gen {gen.max():.0f} ms")
print("PAUSE PROBE DONE")
