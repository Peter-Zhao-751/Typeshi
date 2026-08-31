# Runbook — per-window labels + revision rebalance

Prepared 2026-08-14 on the laptop. Everything below the horizon line is code
that is written, tested (334 green) and ready; this file is the sequence to
run when a GPU is next available, and what to check at each step.

## Why

`build_examples` attached the **session's** labels to **every window**. Over
the 24,909 exported composition windows the `<REV:>` bin matched what its own
window actually did only **52.9%** of the time: 39.9% of windows labelled
`<REV:0>` contained cursor ops, and 24.0% labelled `<REV:n>`, n>0, contained
none. The correct lesson from a token that is wrong a third of the time is to
ignore it, which is what the model does.

The control is in the existing numbers: composition WPM fidelity is r=0.43
against transcription's r=0.994, through the same code. Aalto sentences are
~43 characters, so a transcription session is ~50 events and **never
windows** — its labels were always right. Composition windows heavily. That
predicts every knob is weak in composition and strong in transcription, which
is exactly what `eval_report_composition.json` shows.

## What changed

- `typeshi.labels.window_labels(events, session)` — speed, correction rate
  and revision rate computed from the window; `uncorrected_error_rate`
  inherited, because it measures how far the FINAL text lands from the target
  and a window that typed half the text has not erred by stopping.
- `typeshi.dataset.build_examples` calls it per window.
- `typeshi.serialize.rev_bin` / `rev_from_bin` — `<REV:>` moves to a
  **geometric** scale of its own. It shared `pct_bin` with the error knobs but
  not their magnitudes: ECOR genuinely spans 0-30% (14.8% of composition rows
  clamp at the ceiling) while revisions live near 1%. On whole percents the
  MEDIAN composition window (0.391%) landed in bin 0 next to windows that
  never revise, and 90.9% of all windows fell into bins 0-2 — 31 token values
  expressing three states. Geometric bins over 0.1%-30% spread that mass:
  median window → 8, real writers (1.1-1.3%) → 13-14, heavy reviser (8%) → 23.
  Bin 0 stays exactly zero. Same 31 tokens, so no vocabulary change, no
  embedding resize, no retokenization.
- `typeshi.dataset.revision_repeats(prompt, factor, min_bin)` and
  `build_dataset.py --oversample-revisions / --oversample-min-bin`. Train
  split only — duplicating held-out windows would tilt the distribution the
  discriminator gate reads. Default 1, so the export is byte-identical unless
  asked.

Verified end-to-end on a 400-file KLiCKe slice: label accuracy
**52.9% → 100%**, bins occupied **3 → 28 of 31**, and the share of windows
crammed into bins 0-2 **90.9% → 14.6%** (train export, oversampling on).

**This breaks `<REV:>` compatibility with every existing checkpoint.**
`motor-phase2` learned percent semantics, so `<REV:13>` meant 13% to it and
means 1.06% now. In practice the mismatch is near-harmless — the knob barely
conditions anything on that checkpoint, which is the whole reason for this
change — but the new scale only becomes true with `motor-phase3`. Pick
`--oversample-min-bin` on the NEW scale: bin 17 is 2.3%, bin 23 is 7.6%.

The whole-percent residual is gone with it. Under `pct_bin`, 30% of windows
labelled `<REV:0>` still contained a cursor op, because one op in 300 events
rounds to zero. On the geometric scale bin 0 means exactly zero and 0.33%
lands at bin 6.

---

## Sequence

**Update 2026-08-25:** steps 1 and 1b already ran on the laptop — the
exports exist and passed their gates. The GPU session starts at step 2, and
it now trains TWICE: phase-3 (label fix alone) and phase-4 (+ IteraTeR), so
the two changes stay attributable in one rental. Two decoder-side changes
landed since this file was written and are in the working tree: generation
budgets are run-scoped and `run_eval_composition` conditions each window on
the real paired session's per-window labels (`window_label_schedule`).
Direction of every predicted number below is unchanged; magnitudes are not
comparable to `eval_report_composition.json` beyond direction, because the
old eval conditioned on session-constant labels the model was never
trained to obey.

The eval also changed protocol on 2026-08-14 (`3ae1a69`): CV folds now group
by writer, not by pair. Read phase-3/phase-4 discriminator numbers from the
writer-grouped field; the 0.965 (single-shot) and 0.945 (windowed) figures
referenced in this file are pair-grouped upper bounds.

**1. Re-export (CPU, ~the usual full-corpus build time).** ✅ done locally
2026-08-25, gate passed (label accuracy 100.0%).

    python scripts/build_dataset.py \
      --aalto data/raw/aalto --klicke data/raw/klicke \
      --out data/processed_v3 --oversample-revisions 20 --oversample-min-bin 17

**1b. IteraTeR shard (CPU).** ✅ done locally 2026-08-25:
`data/processed_iterater/` — 6,127 examples, label accuracy 100.0%,
28/31 bins, 11.5% of rows at REV>=17 with NO oversampling. Rebuild with:

    python scripts/build_dataset.py \
      --aalto data/raw/NONE --klicke data/raw/NONE \
      --iterater data/raw/iterater --iterater-timing-from data/raw/klicke \
      --out data/processed_iterater

**1c. Phase-4 mix (CPU, seconds).** Concatenate the shard onto v3's train
split; the eval keeps v3's holdout (IteraTeR writers are never scored by
the KLiCKe eval, and writer-hash splitting is stable across exports):

    mkdir -p data/processed_v4
    cat data/processed_v3/train.jsonl data/processed_iterater/train.jsonl \
      > data/processed_v4/train.jsonl
    cp data/processed_v3/test.jsonl data/processed_v3/split.json data/processed_v4/

Check before training — the export is worthless if this does not hold:

    python - <<'PY'
    import json, re
    REV=re.compile(r"<REV:(\d+)>"); OPS=re.compile(r"<CUR:\d+>|<SELDEL:\d+-\d+>")
    from typeshi.serialize import rev_bin as pct  # the scale, not percents
    EV=re.compile(r"<DT:\d+>|<CUR:\d+>|<SELDEL:\d+-\d+>|<BKSP:\d+>|<(?:SPC|NL|TAB|LT|GT|[^<>]):\d+>")
    n=ok=hi=0
    for line in open("data/processed_v3/train.jsonl"):
        d=json.loads(line); p=d["prompt"]
        if not p.startswith("<MODE:C>"): continue
        ev=len([t for t in EV.findall(d["completion"]) if not t.startswith("<DT:")])
        if not ev: continue
        n+=1; lab=int(REV.search(p).group(1))
        ok += pct(len(OPS.findall(d["completion"]))/ev)==lab
        hi += lab>=17
    print(f"label accuracy {100*ok/n:.1f}% (expect 100.0)   REV>=17 share {100*hi/n:.1f}%")
    # also expect ~28 of 31 bins occupied; 3 would mean the scale did not take
    PY

**2. Retrain Phase 2 from the same foundation, so the comparison is clean.**
Continue `motor-full`, exactly as `motor-phase2` was made — the only change
is the data:

    python -m typeshi.train_motor --mode both \
      --data data/processed_v3/train.jsonl \
      --init-adapter checkpoints/motor-full \
      --out checkpoints/motor-phase3 \
      --epochs 3 --batch 4 --accum 8

**3. Measure knob fidelity — this is the whole point.**

    python scripts/run_eval_composition.py \
      --checkpoint checkpoints/motor-phase3 --n 100 \
      --out eval_report_phase3.json

Read in this order:

| Number | Now | What the fix predicts |
| --- | --- | --- |
| composition WPM r | 0.43 | toward transcription's 0.994 |
| error-rate knob r | ~0 | measurably above 0 |
| `cursor_count` KL | 7.10 | down |
| `seldel_count` KL | 4.91 | down |
| composition discriminator | 0.965 | down |

**4. Re-run Tier-1 to prove transcription did not regress.** The WPM change
reprices any single-window session that contains backspaces, so this is not a
formality:

    python scripts/run_eval.py --checkpoint checkpoints/motor-phase3 \
      --held-out data/heldout_writers --n 200 --out eval_report_phase3_t1.json

Gate: transcription validity stays at 100% and the writer-grouped model gate
stays inside [0.40, 0.55]; motor-full's corrected baseline is 0.5129
(`eval_report_motor-full_writergrouped.json`). The old 0.600 was a
pair-grouped artifact; do not compare against it.

**5. Phase-4: the IteraTeR mix, same rental.** Train from the SAME
foundation with the mixed data, then run both evals again:

    python -m typeshi.train_motor --mode both \
      --data data/processed_v4/train.jsonl \
      --init-adapter checkpoints/motor-full \
      --out checkpoints/motor-phase4 \
      --epochs 3 --batch 4 --accum 8

    python scripts/run_eval_composition.py \
      --checkpoint checkpoints/motor-phase4 --n 100 \
      --out eval_report_phase4.json
    python scripts/run_eval.py --checkpoint checkpoints/motor-phase4 \
      --held-out data/heldout_writers --n 200 --out eval_report_phase4_t1.json

Read phase-4 against phase-3, not against phase-2: the delta IS the
IteraTeR contribution. What it should move that phase-3 alone cannot:
whether generated revisions look like drafts (the discriminator's residual
edge once the count statistics close), and the `cursor_count`/`seldel_count`
distributions at high `<REV:>`. If phase-3 already collapses the revision
KLs and the discriminator, phase-4 is confirmation; if phase-3 helps the
knobs but the discriminator holds at ~0.9 on revision content, phase-4 is
the fix — that fork is the experiment.

**6. Interactive teeth check (either checkpoint, 10 minutes).** The number
the whole thread started from: in the playground, a `<REV:>` around bin
13-17 on a 150-300 char target should now produce visible mid-stream
rewrites — type a phrase, pause, caret back, SELDEL, retype — not only
typo fixes. If the streams still only fix typos while step 3's numbers
look good, say so in the chronicle; that disagreement would mean the eval
is measuring something the eye does not see, and the eye is the customer.

## How this can disappoint

If knob fidelity stays weak after step 3, the label noise was not the binding
constraint and the answer is data volume, not data quality — 218 windows at
`REV>=5` oversampled 20× is still 218 distinct windows, and the model may
simply be memorising them. The tell is train/eval divergence on the
revision statistics specifically. That is the point at which IteraTeR
(`docs/iterater-notes.md`: 4,018 executable human edits, Apache-2.0) earns
its keep, and it needs a synthesis step — turning text edit-actions into
CUR/SELDEL/key sequences with timing drawn from the KLiCKe marginals, which
are the part already nearly solved (hold KL 0.008, pause 0.026).

Do **not** reach for forcing cursor ops in the decoder. It would drive
`cursor_count` KL down while leaving the positions and timing wrong, and the
discriminator would simply catch that instead — the same shape of mistake as
the codec-quantization artifact.
