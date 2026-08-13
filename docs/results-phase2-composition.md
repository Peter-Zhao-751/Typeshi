# Results — Phase-2 composition fine-tune (2026-08-13)

Continued from `checkpoints/motor-full` via `--init-adapter` on the mixed
curriculum (27,480 KLiCKe composition + 32,878 Aalto transcription examples,
`data/processed_phase2`, one writer split across both corpora). 3 epochs,
12h33m, loss 2.94 → 2.569, token accuracy 21.6% (highest of any run).
Checkpoint: `checkpoints/motor-phase2`. Probes: `scripts/probe_phase2.py`.

## Probe results (no Tier-2 eval exists yet — these are sanity probes)

**Transcription regression: 20/20 valid** through the constrained decoder.
The anti-forgetting mix preserved the motor skill outright.

**Composition (5 held-out KLiCKe prompts, UNCONSTRAINED generation):**

- All 5 streams parse — zero grammar failures without any mask, against the
  0.8B shakedown's 64% grammar-failure rate.
- Event mixes track the real sessions per-prompt, including the conditioning:
  a low-revision prompt (real 5.7% BKSP) drew 4.9% from the model; a heavy
  one (20.9%) drew 24.2%. CURSOR (0.4–0.8%) and SELDEL (0.8%) both appear.
- Think-pauses present but thin: ~50–70% of the real >1s fraction per
  session — the same tail-thinness Phase 1 measured.
- It produces realistic typo-revision texture: "To begin gin," (doubled
  syllable, then a revision region), uncorrected slips like "cando".

**The two defects, both decoding-side:**

1. **No convergence pressure**: 2/5 sessions started exactly on target,
   1/5 on-target with heavy typos, and free generation composes its OWN
   on-topic essay when probability drifts -- KLiCKe taught "write an essay
   like this", and nothing at sampling time forces the given text.
2. **Unvalidated cursor ops**: 2/5 sessions emitted a cursor move outside
   the live buffer ("cursor 10 outside buffer of length 1"), which replay
   correctly rejects.

Both are the same missing piece: the design spec §6 buffer-tracking
constrained decoder (on-path masking toward the target, budgeted excursions,
cursor/SELDEL masked to valid positions, EOS legal only when buffer ==
target). The model side of Phase 2 behaves; the guarantee is a decoder, and
that decoder is the next block.

## Licensing

This checkpoint is KLiCKe-derived: **do not publish** until KLiCKe's terms
are resolved.
