"""Does the eval's pair-grouped CV leak writer identity?

Claim under test (from the local workstream): Aalto gives each typist ~15
sessions, so pair-grouped folds put the SAME writer in train and test. Real
sessions carry that writer's fingerprint; generated ones cannot. The
classifier can then score "is this a writer I know" as a proxy for "is this
real" -- inflating accuracy without detecting anything about realism.

Writer-grouped folds are strictly coarser than pair-grouped (a pair's real
and fake both belong to one writer), so they close the twin leak the
pairing was introduced for AND the identity leak.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from typeshi.adapters import aalto
from typeshi.events import Event, EventType
from typeshi.eval.discriminator import featurize, heuristic_baseline, shuffle_timing
from typeshi.labels import compute_labels
from typeshi.serialize import codec_roundtrip


def to_events(rows):
    return [Event(type=EventType[t], char=c, press_time=p, release_time=r)
            for t, c, p, r in rows]


def key(events):
    """Identity of a session: chars + press times are unique in practice."""
    return tuple((e.type.name, e.char, e.press_time) for e in events[:12])


dump = [json.loads(l) for l in open("data/generation_dump_e3.jsonl")]
print(f"{len(dump)} dumped (real, fake) pairs")

# Recover writer IDs by matching dumped real streams back to the corpus.
test_writers = set(json.load(open("checkpoints/motor/split.json"))["test_writers"])
index = {}
for writer, target, events in aalto.iter_sessions("data/heldout_writers"):
    if f"aalto:{writer}" not in test_writers:
        continue
    index.setdefault(key(events), (writer, target))

pairs = []
for d in dump:
    real = to_events(d["real"])
    hit = index.get(key(real))
    if hit is None:
        continue
    writer, target = hit
    pairs.append((writer, target, real, to_events(d["fake"])))
print(f"recovered writer IDs for {len(pairs)}/{len(dump)} pairs "
      f"({len(set(w for w, _, _, _ in pairs))} distinct writers)")

real_s = [codec_roundtrip(r) for _, _, r, _ in pairs]
fake_s = [codec_roundtrip(f) for _, _, _, f in pairs]
writers = [w for w, _, _, _ in pairs]
uw = {w: i for i, w in enumerate(sorted(set(writers)))}
wid = np.array([uw[w] for w in writers])

sess_per_writer = np.bincount(wid)
print(f"sessions per writer: mean {sess_per_writer.mean():.1f}, "
      f"max {sess_per_writer.max()}")


def score(a, b, groups, seed=0):
    X = np.vstack([featurize(s) for s in a] + [featurize(s) for s in b])
    y = np.array([1] * len(a) + [0] * len(b))
    g = np.concatenate([groups, groups])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = GradientBoostingClassifier(random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, groups=g,
                                 scoring="accuracy").mean())


pair_groups = np.arange(len(pairs))
print("\n--- real vs MODEL ---")
print(f"  pair-grouped   (current eval): {score(real_s, fake_s, pair_groups):.4f}")
print(f"  writer-grouped              : {score(real_s, fake_s, wid):.4f}")

# Teeth under the SAME writer-grouped protocol: is it still a real judge?
base = [codec_roundtrip(heuristic_baseline(t, wpm=compute_labels(r, t).wpm or 60,
                                           seed=i))
        for i, (_, t, r, _) in enumerate(pairs)]
shuf = [shuffle_timing(s, seed=i) for i, s in enumerate(real_s)]
print("\n--- teeth under writer-grouped folds ---")
print(f"  vs heuristic baseline: {score(real_s, base, wid):.4f}  (needs >= 0.90)")
print(f"  vs timing-shuffled   : {score(real_s, shuf, wid):.4f}  (needs >= 0.75)")

# Direct test of the mechanism: do these features identify the TYPIST?
keep = [w for w in range(len(sess_per_writer)) if sess_per_writer[w] >= 3]
mask = np.isin(wid, keep)
if mask.sum() > 30 and len(keep) > 5:
    from sklearn.model_selection import StratifiedKFold
    Xi = np.vstack([featurize(s) for s, m in zip(real_s, mask) if m])
    yi = wid[mask]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    acc = float(cross_val_score(GradientBoostingClassifier(random_state=0),
                                Xi, yi, cv=cv, scoring="accuracy").mean())
    print(f"\n--- writer identification from the same features ---")
    print(f"  {len(keep)} writers, {mask.sum()} real sessions: {acc:.4f} "
          f"(chance {1/len(keep):.4f})")
print("\nPROTOCOL TEST DONE")
