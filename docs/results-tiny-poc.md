# Tiny Motor Model PoC — Results

**Date:** 2026-08-11
**Spec:** `docs/superpowers/specs/2026-08-10-tiny-motor-poc-design.md`
**Plan:** `docs/superpowers/plans/2026-08-10-tiny-motor-poc.md`
**Checkpoint:** `checkpoints/motor-tiny` · **Report:** `eval_tiny_full.json`

> **Protocol correction (2026-08-14, commit `3ae1a69`):** every
> model-vs-real accuracy in this file (0.470, 0.610, 0.640) was scored with
> pair-grouped CV folds, which leak writer identity and inflate accuracy
> (measured on the 4B: 0.632 pair-grouped vs 0.518 writer-grouped). The
> stretch-gate verdict is superseded; read the §3 fidelity/realism trade-off
> as directional, not calibrated. The hard-bar results stand, and §5.2's
> unpassable serial gate was closed the same week by the nine order
> statistics (`ec8a89f`: real-vs-shuffled 0.4975 → 0.8075).

A 19M-parameter transformer trained from scratch on one Mac types
transcription sessions that are **100% valid** on held-out writers. The PoC's
hard bar passed; the timing-realism stretch goal did not, and the reason is
now measured rather than guessed.

## 1. Verdict

| gate | value | bar | |
|---|---|---|---|
| `pass_generation_validity` | **1.000** (200/200) | ≥ 0.90 | ✅ |
| `pass_discriminator_has_teeth` | 0.995 | ≥ 0.90 | ✅ |
| `pass_control_near_chance` | 0.445 | [0.40, 0.60] | ✅ |
| `pass_model` (stretch) | 0.640 | [0.40, 0.55] | ❌ |
| `pass_serial_dependence_teeth` (stretch) | 0.500 | ≥ 0.75 | ❌ |

**Hard bar 3/3.** `tier1_met` is False on the two stretch gates only — and one
of those two is unpassable by construction (§5).

Zero malformed generations and zero wrong-text rejections across 200 attempts:
grammar-constrained decoding plus a converged copy circuit leaves no failures
of either kind.

## 2. Training

| | |
|---|---|
| architecture | `Qwen2ForCausalLM` from scratch, 19.13M params (hidden 384, 8 layers, 6 heads MHA, tied embeddings) |
| vocabulary | 12,909 — 97 chars + `<EOS>`/`<PAD>` + 12,810 grammar tokens |
| data | 1,950,110 transcription examples (full Aalto export), 1 epoch |
| wall clock | **7h50m** on an M5 Pro, fp32 on MPS, 69.17 examples/sec |
| loss | 9.29 → **2.549** (token accuracy 1.0% → 19.2%) |
| stability | 276M tokens, zero NaN / OOM / crash; 10 checkpoint probes all healthy |

## 3. The data-scaling curve

Every number below was re-measured with the corrected tokenizer loader (§6.2),
so the curve is honest end to end.

| training data | valid | malformed | wrong-text | model-vs-real |
|---|---|---|---|---|
| 25k × 1 epoch | 0% | 0 | 150 | — |
| 100k × 1 epoch | 77% | 0 | 15 | **0.470** ✅ |
| 100k × 2 epochs | 98% | 0 | 1 | 0.610 |
| **1.95M × 1 epoch** | **100%** | 0 | 0 | 0.640 |

Target-copying is a pure data-volume skill: 0% → 77% → 100%. The 0.8B
pretrained baseline reached 14.2% on 17.7k examples
(`results-08b-shakedown.md`); a 42× smaller model with 110× the data reaches
100%.

**The trade-off in the last column is the run's most interesting finding.**
Text fidelity and timing realism move in opposite directions: the 100k model
that typed only 77% of its targets correctly had timing a discriminator could
*not* separate from real humans (0.470, inside the pass band), while the fully
converged model types perfectly and is separable at 0.640. More training
sharpens the timing distribution toward its mean, and the mean is more
regular than any individual human.

## 4. What the model does

At T=1.0 on held-out targets, character-exact with human-plausible mechanics:

```
target : Northern forces had captured a southern mechanised brigade.
typed  : Northern forces had captured a southern mechanised brigade.
         67 events in 13.8s   (the real human: 67 events in 13.8s)
tokens : <N:55><DT:55><o:52><DT:54><r:50><DT:60><t:49><DT:45>...
```

Rollover is real, not simulated: in one sample `S` releases at 174.6 ms while
`u` presses at 159.3 ms — two keys physically down at once, which is exactly
what the shared bin scale in `docs/token-format.md` was designed to express.

**Knob fidelity** (design spec §8 eval target 3), measured through the
playground API on held-out text:

| requested | realized | | requested | observed |
|---|---|---|---|---|
| 30 WPM | 30.6 | | ECOR 2% | 1 backspace |
| 60 WPM | 65.3 | | ECOR 25% | 13 backspaces |
| 110 WPM | 114.6 | | | |

The conditioning learned from real between-typist variation transfers cleanly
to control at inference.

**Distributional fit** vs held-out writers (KL):

| iki | hold | burst | pause |
|---|---|---|---|
| 0.015 | 0.059 | 0.090 | **0.489** |

Inter-key intervals are nearly perfect. Pauses are the outlier and the likely
source of the 0.640 — the model under-models long thinking pauses, which is
unsurprising for transcription-only training at 19M parameters.

## 5. Two harness defects, both found before they could mislead

### 5.1 The realism gate was unpassable for *any* generator

Generations exit `deserialize()` carrying bin-center timings; real sessions
carried raw milliseconds. Measured with `scripts/harness_control.py`: a
discriminator separates real sessions from **their own round-trips** at
**0.915** (0.910 timing-only). Quantization alone was the signal.

Fixed by symmetrizing (`run_eval.py` now round-trips real and baseline
sessions through the same path before featurizing). Every number in this
document is post-fix. **The 0.8B's 0.77 was substantially this artifact.**

### 5.2 The serial-dependence gate cannot pass at all

`pass_serial_dependence_teeth` asks the discriminator to separate real
sessions from timing-shuffled real sessions (≥ 0.75). Measured on 120
held-out sessions (`scripts/shuffle_diagnostic.py --n 120`,
`shuffle_diagnostic.json`):

| input | real-vs-shuffled |
|---|---|
| raw milliseconds | 0.542 |
| round-tripped | 0.483 |

Neither is close, and the fix in §5.1 is not the cause. **8 of `featurize`'s
33 features are order-sensitive, not 1**: the 7 burst-block stats (5
quantiles + mean + std of run lengths, indices 24–30 — `distributional.py`
already splits bursts on any inter-key gap > 1000 ms) plus lag-1 log-IKI
autocorrelation. Shuffling moves where those pauses fall, so it moves burst
run-lengths too; the burst *count* alone is invariant, since permuting gaps
doesn't change how many exceed the threshold. The other 25 features are
marginal statistics, identical by construction.

The real reason the gate cannot pass: long pauses are rare in Aalto
transcription (short single sentences), so per-session burst run-lengths are
near-degenerate — usually one burst covering the whole session. Measured
directly: shuffling changes the burst features in only **38 of 120 sessions
(32%)**, so in the other 68% those seven features are shuffle-invariant too.
And **lag-1 log-IKI autocorrelation of real Aalto sessions is +0.009**, i.e.
zero. Between them the order-sensitive features carry almost no per-session
signal, so the gate is unpassable regardless of model.

**This affects the Phase-1 GPU eval identically.** Closing it needs
order-sensitive features that actually carry signal and burst run-length
statistics alone are not enough — they are already implemented
(`distributional.py`'s burst block, above) and still don't clear the bar.
Better candidates: multi-lag autocorrelation, digraph-conditioned timing
deviations, or a sequence model over the raw event stream (the design spec's
original "Transformer over event streams" discriminator,
`docs/superpowers/specs/2026-08-09-typing-process-model-design.md` §Discriminator
Turing test). That's a Phase-1 task, not a threshold tweak.

## 6. Bugs this PoC found in shared code

Three of the four affect the Phase-1 GPU run and are already fixed on this
branch.

1. **EOS trained where decoding forbade it** (`constrain.py`). TRL appends EOS
   directly after the completion's final event token — a *gap* slot — but the
   grammar mask only legalized EOS in *event* slots. Every model was trained to
   emit EOS exactly where the mask forbade it; a pretrained model limps past
   this, a from-scratch one cannot terminate at all. **Phase 1 affected.**
2. **`AutoTokenizer` silently ate every space** (§6.2 below). **Phase 1 at risk**
   if its checkpoint layout ever shifts; the new guard covers both paths.
3. **The eval crashed at 1–4 valid pairs** — `StratifiedGroupKFold(n_splits=5)`
   needs five per class. Zero valid reported honestly, 1–4 died after doing all
   the generation work. Any early checkpoint hits this window. **Phase 1 affected.**
4. **Unsupported characters killed the playground's request thread** — the
   tokenizer raises a pyo3 exception, not `ValueError`. Now pre-checked with a
   named-offender message.

### 6.2 The one that cost a night's conclusion

`AutoTokenizer.from_pretrained(<tiny checkpoint>)` cannot resolve the saved
generic wrapper class, falls back to `config.json`'s `model_type: "qwen2"`, and
instantiates `Qwen2Tokenizer` — whose byte-level handling **drops every literal
space at encode time** (7 spaces in a prompt → 0 space tokens). The model was
faithfully typing the spaceless prompts it was handed.

This made the 100k pilot read **0/150 valid** when the same checkpoint actually
scores 98%. The pilot's stop condition fired on an artifact, and the "the model
can't bind spaces" theory built on top of it was wrong.

The fix (`typeshi/eval/load.py`) routes plain checkpoints to the generic class
and — more importantly — **refuses to run any eval unless the loaded tokenizer
proves a byte-exact round-trip** on a probe containing spaces and adjacent
grammar tokens. Failing loud costs seconds; failing quiet cost a pilot cycle.

## 7. Temperature is not the missing lever

The spec named temperature as the first lever if realism lagged. It is
exhausted, and the answer is no:

| T | valid | model-vs-real | pause KL |
|---|---|---|---|
| **1.00** | 100% | **0.640** | **0.489** |
| 1.10 | 100% | 0.645 | 1.828 |
| 1.20 | 98% | 0.710 | 2.410 |
| 1.35 | 88% | 0.740 | 1.447 |

`model-vs-real` degrades monotonically with higher temperature while validity
erodes; pause KL does not move monotonically but blows up, peaking at ~5× (T=1.2)
before falling back at T=1.35. Either way it widens the distribution in the
wrong places. **T=1.0 is optimal.**
The remaining timing gap is capacity and training distribution, not sampling
sharpness.

## 8. What this proves, and what it does not

**Proven.** The token format is learnable end to end; the dataset export,
constrained decoder, and eval plumbing work against a second independent
model and tokenizer; target-copying is a data-volume skill that saturates at
100%; the knobs steer real behaviour; the machine can train the whole pipeline
overnight without a GPU rental.

**Not proven — still open for the GPU run.**

1. **BPE-target → char copying.** The 0.8B's dominant failure (103/120
   wrong-text) is exactly the skill the char-level tokenizer engineers away.
   This PoC does not discharge that risk; the 7B pilot's first checkpoint gate
   should be wrong-text rate specifically.
2. **Timing realism at capacity.** 19M types correctly but not
   indistinguishably. Whether 7B closes the 0.640 gap — and whether the §3
   fidelity/realism trade-off reappears at scale — is untested.
3. Added-token registration over a 152k vocabulary, embedding seeding, LoRA
   `modules_to_save`, and bf16 CUDA numerics remain exercised only by the
   Phase-1 path.
4. **Composition mode** — revisions, cursor movement, `<CUR:>`/`<SELDEL:>` —
   is entirely out of scope here.

## 9. Try it

```bash
uv run python scripts/playground.py --device mps    # http://localhost:8765
```

Type any ASCII text, set WPM / error rates / temperature / seed, and watch the
model type it in real time on a simulated keyboard — each key lit for exactly
its hold duration, overlapping presses included. `scripts/watch_training.py`
is the live dashboard for training runs.
