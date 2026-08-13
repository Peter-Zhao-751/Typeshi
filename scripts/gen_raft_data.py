"""Rejection-sampling (RAFT) preference data for the Phase-3 polish.

The Tier-1 model gate plateaued at 0.58-0.60 after data, epochs, and
temperature were exhausted; the design spec's remaining lever (section 5,
phase 3) is discriminator-guided preference optimization. This script is
the data half: K candidate generations per TRAINING-writer target, scored
by a discriminator trained on TRAINING-writer real sessions vs model
generations, keeping the most-human candidate per target as SFT data.

Held-out writers appear nowhere here: the scoring discriminator and every
target come from the training split, so the eval stays uncontaminated.

Output: {"prompt", "completion"} JSONL compatible with train_motor
--init-adapter (the RAFT step is plain SFT on winners -- no new trainer).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from typeshi.eval.discriminator import featurize, train_discriminator
from typeshi.generate import generate
from typeshi.labels import SessionLabels
from typeshi.serialize import codec_roundtrip, serialize
from typeshi.buffer import replay
from typeshi.labels import _levenshtein


def parse_prompt(prompt: str):
    m = re.search(r"<TARGET>(.*)<PROCESS>", prompt, re.S)
    return m.group(1) if m else None


def labels_from_prompt(prompt: str) -> SessionLabels | None:
    """SessionLabels whose binning reproduces the stored prompt EXACTLY.

    generate() rebuilds its prompt from labels, so the winner must be
    sampled under the stored prompt's own knob bins -- anything else pairs a
    completion with conditioning it was not generated under and trains the
    model that the knobs are noise. wpm_bin(b*5) == b and pct_bin(b/100)
    == b invert the binning; the caller asserts byte-equality.
    """
    m = re.search(r"<WPM:(\d+)><ECOR:(\d+)><EUNC:(\d+)><REV:(\d+)>", prompt)
    if not m:
        return None
    w, ec, eu, rv = (int(g) for g in m.groups())
    return SessionLabels(w * 5, ec / 100, eu / 100, rv / 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/motor-full"))
    ap.add_argument("--train-data", type=Path,
                    default=Path("data/processed_full/train.jsonl"))
    ap.add_argument("--targets", type=int, default=800)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("data/raft/train.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from typeshi.eval.load import load_checkpoint_model, load_checkpoint_tokenizer
    from typeshi.serialize import deserialize
    from typeshi.train_motor import _detect_backend

    tok = load_checkpoint_tokenizer(args.checkpoint)
    model = load_checkpoint_model(args.checkpoint, _detect_backend())
    model.eval()

    # Unique targets + their real completions, in file order (train split).
    rng = np.random.default_rng(args.seed)
    rows = []
    seen = set()
    with args.train_data.open() as fh:
        for line in fh:
            ex = json.loads(line)
            tgt = parse_prompt(ex["prompt"])
            if not tgt or tgt in seen or "<MODE:T>" not in ex["prompt"]:
                continue
            seen.add(tgt)
            rows.append(ex)
    order = rng.permutation(len(rows))[: args.targets]
    picked = [rows[i] for i in order]
    print(f"{len(picked)} training targets", flush=True)

    # Candidate generation, K per target, valid ones only.
    real_events = [deserialize(ex["completion"]) for ex in picked]
    winners = []
    gen_pool = []   # (target_idx, events) for discriminator training
    skipped_labels = 0
    for ti, ex in enumerate(picked):
        tgt = parse_prompt(ex["prompt"])
        labels = labels_from_prompt(ex["prompt"])
        from typeshi.dataset import build_prompt

        if labels is None or build_prompt(tgt, labels, "transcription") != ex["prompt"]:
            skipped_labels += 1
            continue
        cands = []
        for k in range(args.k):
            try:
                ev = generate(model, tok, tgt, labels, mode="transcription",
                              temperature=1.0, max_new_tokens=4 * len(tgt) + 64,
                              seed=args.seed + ti * args.k + k)
            except ValueError:
                continue
            sim = 1 - _levenshtein(replay(ev), tgt) / max(len(tgt), 1)
            if sim >= 0.8:
                cands.append(ev)
        if cands:
            winners.append((ti, ex, cands))
            gen_pool.extend((ti, c) for c in cands)
        if (ti + 1) % 50 == 0:
            print(f"  {ti+1}/{len(picked)} targets, {len(gen_pool)} candidates",
                  flush=True)

    # Score: discriminator on training-real vs candidates (codec-fair).
    rt_real = [codec_roundtrip(real_events[ti]) for ti, _, _ in winners]
    rt_fake = [codec_roundtrip(cs[0]) for _, _, cs in winners]
    clf, acc = train_discriminator(rt_real, rt_fake, paired=True)
    print(f"scoring discriminator paired CV (train writers): {acc:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.out.open("w") as out:
        for ti, ex, cands in winners:
            feats = np.vstack([featurize(codec_roundtrip(c)) for c in cands])
            p_real = clf.predict_proba(feats)[:, 1]
            best = cands[int(np.argmax(p_real))]
            out.write(json.dumps({
                "prompt": ex["prompt"],
                "completion": serialize(best),
            }) + "\n")
            kept += 1
    print(f"wrote {kept} winner examples to {args.out} "
          f"(skipped {skipped_labels} non-invertible prompts)", flush=True)
    print("RAFT DATA DONE", flush=True)


if __name__ == "__main__":
    main()
