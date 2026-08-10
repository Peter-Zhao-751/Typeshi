# 0.8B Local Shakedown — Results

First end-to-end exercise of the full loop (train → checkpoint → five-gate
eval) on real hardware. Qwen3.5-0.8B, one epoch over 3,846 transcription
examples, fp32 on an M5 Pro via MPS. **This was a pipeline shakedown, not a
serious training run** — 4k examples is ~0.2% of the corpus.

## Training

| | |
|---|---|
| loss | 5.53 → 4.18, still falling at epoch end |
| wall clock | 26 min (108 steps, 2.2 examples/s) |
| trainable | 275M (embeddings + 60 LoRA matrices, all 24 layers) |

## Eval (`eval_report_08b.json`)

120 attempts against 29 held-out Aalto writers, `max_new_tokens=512`:

| outcome | count |
|---|---|
| valid (parses + types the target) | **0** |
| malformed grammar | 77 (64%) |
| parsed but wrong text | 42 (35%) |

All five gates failed; `tier1_met: false`. This is the honest expected result.

## Failure modes, from raw generations

1. **The grammar is essentially learned.** Generations open with dozens of
   perfectly alternating `<c:h><DT:k>` pairs, holds in bins 44–53 and DTs in
   43–74 — realistic regions of the timing scale. One 26-minute epoch was
   enough for format.
2. **Sampling falls off the token island.** Mid-stream, sampled runs derail
   into base-vocabulary text (Thai casino spam, memorably). The 12.8k event
   tokens are a small region of a 248k vocabulary; once one off-island token
   is sampled the sequence cascades away and never returns. Greedy decoding
   never derails — this is a probability-mass problem, not a structural one.
3. **Target conditioning is not learned yet.** Greedy types plausible
   character rhythms (`te te te ...`) unrelated to the prompt's target
   sentence. Copying-the-target is the harder skill and needs far more data.

## What this implies

- **More training data is the first lever** (modes 2 and 3 are both
  undertraining). The corpus has 1.99M transcription examples; this run used
  0.2% for one epoch.
- **Constrained decoding — already planned in the design spec — eliminates
  mode 2 by construction** (mask logits to grammar-legal tokens). This run is
  direct evidence for why the spec wants it; worth prioritising when Phase-1
  training is otherwise healthy.
- The eval harness held up: bounded attempts, honest zero-valid report,
  per-mode failure counts. Three eval-runner bugs were found and fixed during
  this shakedown (unbounded generation budget, unbounded attempt sweep,
  no-report-on-zero-valid) — which is what shakedowns are for.
