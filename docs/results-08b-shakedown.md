# 0.8B Local Shakedown — Results

> **Verdicts superseded.** The 0.77 model-vs-real figure below is inflated
> twice over: real sessions were scored raw against bin-quantized
> generations (the codec comb, `results-tiny-poc.md` §5.1), and the CV folds
> were pair-grouped, leaking writer identity (`3ae1a69`, 2026-08-14). Tier-1
> has since been met by the 4B checkpoints under the corrected protocol
> (`docs/results-qwen35-4b-gpu.md`). The shakedown narrative, the MPS
> sampler bug, and the data-scaling curve stand.

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

---

# Overnight Run — Corrected Results (2026-08-10)

Qwen3.5-0.8B, 3 epochs × 17,714 examples (1,204 writers), 6h17m on MPS.
Loss 5.49 → 2.73 (most of the gain inside epoch 1: data, not repetition —
epochs 2–3 only moved 2.91 → 2.73). Token accuracy 16.8%.

## The sampler discovery

The morning eval's impossible "malformed under the mask" counts led to
catching **torch.multinomial on MPS emitting a token whose logit was a
verified -inf** (~1 violation per 250 steps). Every previous eval number was
corrupted by this: the off-vocabulary text attributed to "probability mass"
in the first shakedown was this kernel bug all along. Sampling is now
Gumbel-argmax (distributionally identical, argmax-only, device-independent,
seeded on CPU) — parse rate went 27% → 12/12 on the same checkpoint.

## Eval with the trustworthy sampler (17 valid pairs — small-N caveats apply)

| gate | value | verdict |
|---|---|---|
| teeth vs naive baseline | 0.97 | ✅ |
| control (real vs real) | 0.47 | ✅ |
| serial-dependence teeth | 0.54 | ❌ (needs 0.75; n=17 limits the discriminator itself) |
| model vs real | 0.77 | ❌ (needs ≤0.55 — timing is genuinely distinguishable) |
| generation validity | **14.2%** (17/120, **0 malformed**, 103 wrong-text) | ❌ (needs 90%) |

## The data-scaling curve so far

| training data | valid generations |
|---|---|
| 3.8k examples, 1 epoch | 0/120 |
| 17.7k examples, 3 epochs | **17/120 (14.2%)** |

Grammar is solved by construction (constrained decoding: zero malformed).
The binding constraint is now **target fidelity** — generations type related
but imperfect text ("imablance" transpositions, dropped words). That is a
data-volume skill; the full corpus is 112× the overnight set. Timing realism
(0.77 discriminator) is the second gap and also expected to improve with
data; if it does not, the levers are temperature and longer training before
recipe changes.
