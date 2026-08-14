# Results — Qwen3.5-4B GPU run (2026-08-11)

Phase-1 motor fine-tune on a Lambda 1×H100 80GB. Base model Qwen3.5-4B (run
owner's direction, superseding the runbook's Qwen2.5-7B default). Training
data: the 20k-file Aalto subset (236,489 train examples), writer-split with
the subset-stable hash split.

## Verdict

**Tier-1 IS MET.** Both `motor-full` and `motor-phase2` pass all five gates
under the corrected evaluation protocol (2026-08-14,
`eval_report_motor-full_writergrouped.json`): model 0.5129 / 0.5100,
validity 1.000 / 0.995, teeth 0.997 / 1.000, serial teeth 0.788 / 0.758,
controls 0.470 / 0.500.

Everything below this section was written while the gate appeared to fail,
and its numbers are pair-grouped — inflated by the writer-identity leak
described at the end. The narrative is left intact because the levers it
records were genuinely spent, but **its verdicts are superseded**: the model
was never the problem.

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

## Full corpus, and the lever exhaustion (2026-08-12)

One epoch on all 1.95M examples (20h23m, loss 2.601,
`checkpoints/motor-full`, `eval_report_full.json`): marginals improved
across the board — pause KL 0.92 → 0.46, iki 0.021 → 0.014, burst
0.100 → 0.067, validity a perfect 200/200 — **and the gate did not move**:
0.5775 → 0.60, flat within CV noise on the same held-out pool. A
temperature sweep then closed the last cheap hypothesis: 1.0 → 0.60,
0.9 → 0.66, 0.8 → 0.685 — monotone, cooling pulls generation variance
below human variance and the discriminator reads exactly that.

Phase-1 lever ledger:

| Lever | Result |
|---|---|
| More data (8.25×) | marginals ↑, gate flat |
| Longer training (3 epochs) | gate flat |
| Lower temperature | actively harmful, monotone |
| Stronger eval (serial features, codec-fair) | made the gate honest |

Tier-1 stands at 4/5 with `pass_model` ~0.58–0.60 against ≤0.55. The
remaining designed route is Phase-3 adversarial polish
(discriminator-guided preference optimization); the residual signal is
diffuse sequential structure (top GBM importance 0.092, hold/gap coupling),
which none of the sampling- or scale-side levers touch.

Infrastructure notes for that run: training peaks at 24.1 GiB (batch 32);
`flash-linear-attention` + `causal-conv1d` must both import or decode drops
~2× and the whole fast path disarms (causal-conv1d needs a source build
against cuda-toolkit-13-0 — no wheel for torch 2.13); the parallel dataset
builder rebuilds the full corpus in ~3 min.


## Correction (2026-08-14): the gate was measuring writer identity

Every `pass_model` number above is wrong, in the same direction, for one
reason. Aalto gives each participant ~15 sessions and `run_eval` walked its
held-out corpus file by file, so 200 "independent" scored sessions came
from **14 writers** — and pair-grouped CV folds then placed the same writer
in train and test. A real session carries that writer's fingerprint; a
generated one cannot. The classifier could therefore score "typist I
recognise" as a proxy for "real", which is not what the gate is asking.

Measured on this box's own generations, 8 seeds:

| protocol | accuracy |
|---|---|
| pair-grouped (all reports above) | 0.6322 ± 0.0120 |
| writer-grouped | **0.5181 ± 0.0085** |

and the same features identify which of the 14 typists wrote a session at
0.595 against 0.071 chance — the leak, directly.

Writer grouping is strictly coarser than pair grouping (a pair's real and
fake share a writer), so it keeps the twin protection pairing existed for
and adds identity protection, and it is not a blunter judge: it still
catches the heuristic baseline at 0.997 and timing-shuffled real sessions
at 0.788. The eval now groups by writer and caps sessions per participant
(`--max-per-writer`, default 3, which took the sample from 14 writers to
67). Commit `3ae1a69`.

The consequence for the lever ledger above: more data, more epochs, lower
temperature and RAFT were all measured against a metric that could not move
for model-quality reasons. Their null results say nothing about those
levers. The one real finding they contain is distributional — the marginals
genuinely improved with data (pause KL 0.92 → 0.46).

Credit: found by the local (Mac) workstream and replicated here before it
was believed.
