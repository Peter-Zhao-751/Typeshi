"""Dump 200 paired real/generated sessions, then ask the discriminator
what separates them: GBM feature importances + per-session stats."""
import json

import numpy as np
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from typeshi.adapters import aalto
from typeshi.eval.discriminator import _QUANTILES, featurize, train_discriminator
from typeshi.generate import generate
from typeshi.labels import compute_labels

ck = "checkpoints/motor"
tok = AutoTokenizer.from_pretrained(ck)
model = AutoPeftModelForCausalLM.from_pretrained(ck, dtype=torch.bfloat16, device_map="auto")
model.eval()
test_writers = set(json.load(open(f"{ck}/split.json"))["test_writers"])

real, fake = [], []
i = 0
for writer, target, events in aalto.iter_sessions("data/heldout_writers"):
    if f"aalto:{writer}" not in test_writers:
        continue
    labels = compute_labels(events, target)
    try:
        gen = generate(model, tok, target, labels, temperature=1.0,
                       max_new_tokens=min(512, 4 * len(target) + 64), seed=i)
    except ValueError:
        continue
    real.append(events)
    fake.append(gen)
    i += 1
    if i >= 200:
        break
print(f"{len(real)} pairs", flush=True)

# persist for later analysis without regeneration
with open("data/generation_dump_e3.jsonl", "w") as fh:
    for r, f in zip(real, fake):
        fh.write(json.dumps({
            "real": [(e.type.name, e.char, e.press_time, e.release_time) for e in r],
            "fake": [(e.type.name, e.char, e.press_time, e.release_time) for e in f],
        }) + "\n")

# feature names must mirror featurize()
names = []
for key in ("iki", "hold", "pause", "burst"):
    names += [f"{key}_q{int(q*100)}" for q in _QUANTILES]
    names += [f"{key}_mean", f"{key}_std", f"{key}_count"]
names += ["lag1", "lag2", "lag3", "von_neumann", "markov_excess",
          "local_spread", "drift", "hold_gap_corr", "word_boundary"]

clf, acc = train_discriminator(real, fake, paired=True, seed=0)
print(f"paired CV accuracy: {acc:.4f}")
vec = featurize(real[0])
assert len(names) == len(vec), f"{len(names)} names vs {len(vec)} features"
imp = clf.feature_importances_
order = np.argsort(imp)[::-1]
print("top separating features:")
for j in order[:12]:
    print(f"  {names[j]:>16} {imp[j]:.3f}")

# per-session distribution comparisons for the top suspects
def session_stats(sessions):
    means, stds, counts = [], [], []
    for s in sessions:
        ikis = np.diff([e.press_time for e in s])
        if ikis.size:
            means.append(np.log1p(ikis).mean())
            stds.append(np.log1p(ikis).std())
            counts.append(len(s))
    return np.array(means), np.array(stds), np.array(counts)

rm, rs, rc = session_stats(real)
fm, fs, fc = session_stats(fake)
print("\nbetween-session spread (std across sessions):")
print(f"  mean log-IKI:  real {rm.std():.4f}   gen {fm.std():.4f}")
print(f"  std log-IKI:   real {rs.std():.4f}   gen {fs.std():.4f}")
print(f"  event count:   real {rc.mean():.1f}+-{rc.std():.1f}   gen {fc.mean():.1f}+-{fc.std():.1f}")
print("ANALYZE DONE", flush=True)
