# Results — Qwen3.5-4B GPU run (2026-08-11)

Phase-1 motor fine-tune on a Lambda 1×H100 80GB. Base model Qwen3.5-4B (run
owner's direction, superseding the runbook's Qwen2.5-7B default). Training
data: the 20k-file Aalto subset (236,489 train examples), writer-split with
the subset-stable hash split.

## Verdict

**Tier-1 not met — 4 of 5 gates pass; `pass_model` misses by 0.0275**
(0.5775 against the ≤0.55 ceiling). Report:
`eval_report_e3_codecfair.json`.

| Gate | Value | Needs | |
|---|---|---|---|
| generation_validity | 99.5% | ≥90% | ✅ |
| discriminator_has_teeth | 1.00 | ≥0.90 | ✅ |
| serial_dependence_teeth | 0.8075 | ≥0.75 | ✅ |
| control_near_chance | 0.465 | 0.40–0.60 | ✅ |
| model | **0.5775** | 0.40–0.55 | ❌ |

Marginals (KL, real vs generated, in-codec): iki 0.021, hold 0.051,
burst 0.100, pause 0.92. The residual discriminator signal is diffuse —
top GBM importance is 0.092 (hold/gap coupling), then hold_std, iki_q50,
lag-2 autocorrelation, word-boundary timing.

## Two bugs this run flushed out

Both would have failed any model, and both are now pinned by offline tests.

**1. EOS was grammar-legal exactly where a stream cannot end** (`d51f757`).
The constrained decoder allowed EOS after `<DT:>` (the dangling-gap ending
deserialize rejects) and forbade it after an event token (the only legal
stop — the final keystroke has no gap). The trained model put p≥0.9998 on
EOS at the true stop; the mask −inf'ed it and forced it to keep typing.
Validity was 1% before the one-line fix, 99.5% after, at every temperature.
It survived because the module's only tests were network-gated and always
skipped.

**2. The discriminator was reading the codec, not the model** (`c0aa902`).
Decoded generations only take ~40 distinct hold values (the 128-bin log
grid); raw corpus sessions take hundreds. The GBM put 0.438 of its
importance on one hold quantile — a quantization comb no token-emitting
model can beat. Scoring both sides in-representation
(`codec_roundtripped_real: true`) moved model accuracy 0.8225 → 0.5775 and
pause KL 7.29 → 0.92. The raw-real number stays in the report.

## Training trajectory

| Checkpoint | Data | Discriminator (codec-fair) | Validity |
|---|---|---|---|
| epoch 1 | 236k ex | — (scored 0.895 raw, old features) | 99.5% |
| epoch 3 | 236k ex ×3 | **0.5775** | 99.5% |

Extra epochs did not fix the pause tail (KL worsened raw); the codec-fair
rescore revealed most of that "gap" was the comb. Loss plateaued ~2.64 by
mid-epoch-2.

The eval discriminator also gained sequential teeth this run (`ec8a89f`):
nine order statistics (autocorrelations, von Neumann ratio, Markov burst
excess, drift, hold/gap coupling, word-boundary slowdown) took
real-vs-shuffled from 0.4975 (chance — the gate could never pass) to 0.8075
on real held-out sessions, with the control at 0.465.

## What's left

The plan's remaining lever is data: `data/processed_full` (1.95M examples,
8.25× the subset, same held-out writers by construction) is built and ready.
At the measured 39.6 samples/s an epoch is ~14h. The 0.0275 gap is small and
diffuse; one full-corpus epoch is the obvious next experiment.

Infrastructure notes for that run: training peaks at 24.1 GiB (batch 32);
`flash-linear-attention` + `causal-conv1d` must both import or decode drops
~2× and the whole fast path disarms (causal-conv1d needs a source build
against cuda-toolkit-13-0 — no wheel for torch 2.13); the parallel dataset
builder rebuilds the full corpus in ~3 min.
