# Typing Process Model — Design

**Date:** 2026-08-09
**Status:** Approved direction (locked in brainstorming); pending user review of this written spec.

## 1. Goal

An end-to-end neural system that takes a finished English text plus control knobs and outputs a
complete, human-indistinguishable **writing process**: a timestamped keystroke event stream that
composes the text the way a real person thinking-as-they-write would — cognitive pauses, clause
bursts, motor typos with corrections, and non-linear **semantic revisions** (typing a different
earlier wording, then revising it), converging exactly on the given final text.

**Success criterion:** a strong learned discriminator (trained on real vs. generated event streams)
cannot beat ~50% accuracy, and realized WPM / error-rate track the requested knob values.

### Non-goals (v1)

- Languages other than English.
- Touchscreen/mobile typing (Aalto mobile data exists if wanted later).
- Real personas / mimicking a specific individual (a generic "style" knob may come later).
- A GUI or browser-automation integration; v1 delivers a library/API that emits the event stream.

## 2. Architecture: one fine-tuned LLM

A single open-weight LLM (~7–8B class, e.g. Llama or Qwen), LoRA-fine-tuned so that:

- **Prompt** = target final text + control knobs (target WPM, error rate, revision propensity),
  encoded as a structured header.
- **Completion** = the serialized writing process as a token stream (keys, cursor operations,
  quantized inter-event times).

Why this shape (over the earlier three-stage pipeline):

- The pretrained LLM already knows English semantics, so "invent a plausible earlier draft of this
  clause, then revise it into the target" comes largely free from pretraining — the fine-tune only
  teaches the output format and the behavioral statistics. This is what makes semantic revisions
  learnable from only ~5k composition logs.
- The softmax over quantized time bins **is** the distributional timing output: sampling gives
  heavy-tailed, non-metronomic timing naturally. No mixture-density head needed.
- It is a single black box, matching the project's intent. No hand-written timing, typo, or pause
  rules anywhere in the model.

## 3. Event serialization

The vocabulary is extended with special tokens:

| Token class | Meaning |
|---|---|
| `<KEY:x>` | Press+release of key `x` (letters, digits, punctuation, space, enter, shift-variants) |
| `<BKSP>` | Backspace |
| `<CURSOR:n>` | Move cursor to buffer position `n` (click or arrow-key navigation collapsed) |
| `<SELDEL:a-b>` | Select range `[a,b)` and delete |
| `<DT:k>` | Inter-event time in log-spaced bin `k` (~128 bins; few-ms resolution at the fast end, seconds-scale at the slow end) |
| `<HOLD:k>` | Key hold duration bin (press→release), enabling rollover when next press precedes release |
| `<EOS>` | Process complete; buffer must equal target text |

Every key event is followed by its `<HOLD:k>` and preceded by a `<DT:k>` gap. Long "thinking"
pauses are just large `DT` bins — no special pause token, so pause behavior is fully learned.

**Session length:** a full essay is tens of thousands of events, exceeding one context window.
Training uses windowed chunks with a rolling context header (recent buffer state + position + knob
values), the standard long-sequence treatment. Inference streams the same way.

## 4. Data

### 4.1 Sources

1. **Aalto 136M Keystrokes** (168k users, desktop transcription, press/release timestamps +
   keycodes, per-session WPM & error stats). Teaches **motor realism**: digraph-dependent
   inter-key intervals, hold times, rollover, typo/correction micro-rhythm.
2. **KLiCKe corpus** (~5k argumentative essays, full keystroke + cursor logs, CSV/IDFX, writer
   demographics + typing skill + quality scores). Teaches **composition behavior**: planning
   pauses, clause bursts, revision patterns, non-linear cursor movement.
3. **Synthetic revision trajectories** (augmentation): backward-constructed draft chains
   `D0 → … → Dn = T` — an LLM writes plausible earlier drafts of arbitrary texts; diffs between
   consecutive drafts become edit-op streams; timings assigned by the motor model once it exists.
   Used only to enrich fine-tuning data, not at runtime.

### 4.2 Preparation

- Parse both corpora into a unified event-stream schema (the serialization above).
- Derive per-session condition labels: realized WPM, uncorrected + corrected error rates,
  revision-op fraction. These become the knob values in each training prompt (so the model learns
  the knob→behavior mapping from real variation across typists).
- Tag each example with a mode token (`<MODE:transcription>` / `<MODE:composition>`); inference
  uses composition mode. Transcription data still transfers motor statistics through shared
  key/timing tokens.
- Holdout: split by **writer**, never by session, for all evaluation.

## 5. Training curriculum

1. **Phase 1 — motor pretraining:** LoRA fine-tune on serialized Aalto transcription sessions
   (millions of short examples). Model learns format + motor timing distributions.
2. **Phase 2 — composition fine-tuning:** continue on KLiCKe full-process examples plus synthetic
   revision trajectories, mixing in a fraction of Aalto to prevent motor forgetting.
3. **Phase 3 (optional, if discriminator eval shows gaps):** adversarial polish — train the
   discriminator, use it for rejection-sampling-based preference data (DPO-style) on generations.

Hardware: one rented H200 (or A100-80GB) covers LoRA fine-tuning of an 8B model comfortably.

## 6. Inference & convergence guarantee

Free-running generation can drift from the target text. Decoding-time guardrail (not a behavioral
rule — all timing and behavioral choices remain sampled from the model):

- Maintain the simulated text buffer during decoding.
- **Constrained decoding** with a custom logits processor: when the model is typing "on-path"
  (buffer is consistent with a prefix/known draft of the target), forward key tokens are masked to
  those consistent with reaching the target; `DT`/`HOLD`/`BKSP`/typo excursions remain free.
- **Bounded excursions:** typos and semantic-revision excursions (off-target wording) are allowed
  up to a budget; the processor requires resolution (correction/revision back toward the target)
  before `<EOS>` becomes legal. `<EOS>` is only unmasked when buffer == target.
- Fallback if needed: plan-first training format (model emits a compact draft-chain sketch before
  the event stream), then rejection sampling as a last resort.

## 7. Control knobs (v1)

| Knob | Range | Mechanism |
|---|---|---|
| Target WPM | ~20–120 | Prompt header value, learned from per-session labels |
| Error rate | corrected + uncorrected % | Prompt header value |
| Revision propensity | low/med/high | Prompt header value (fraction of revision ops in session label) |
| Sampling temperature | fixed default | Exposed for power users; affects variability |

Knob fidelity is an eval target (§8), not assumed.

## 8. Evaluation

1. **Distributional metrics** vs. held-out writers: KL divergence & Fréchet distance on IKI, hold
   time, pause-length, and burst-length distributions (KeyGAN protocol).
2. **Discriminator Turing test:** train a strong classifier (Transformer over event streams) on
   real vs. generated; target ≤ ~55% accuracy. Also test against off-the-shelf heuristic
   simulators (human-keyboard, HumanTyping) as baselines the discriminator *should* beat easily —
   validating the discriminator itself.
3. **Knob fidelity:** sweep requested WPM/error-rate, measure realized values, report correlation
   and calibration error.
4. **Composition signatures:** pause-at-clause-boundary distributions, P-burst length
   distributions, revision-op statistics vs. KLiCKe held-out — the specifically Tier-2 checks.
5. **Convergence:** 100% of generations end with buffer == target (guaranteed by decoding; verify).

## 9. Serving

- **Primary:** Modal or Baseten (serverless GPU, scale-to-zero) running vLLM with the custom
  constrained-decoding logits processor. Chosen because the convergence guardrail requires custom
  decoding logic that catalog serverless LoRA APIs (Fireworks/Together) cannot run.
- **Fallback/cheap tier:** if unconstrained sampling proves reliable enough in practice,
  Fireworks/Together serverless LoRA (pay-per-token, no idle cost).
- Output API: JSON event stream `[{event, key?, pos?, t_press, t_release?}, …]` — consumers replay
  it however they like.

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Constrained decoding distorts timing statistics near constraints | Measure discriminator score with/without constraints; loosen via plan-first format if needed |
| KLiCKe too small for revision diversity | Synthetic backward-chain augmentation (§4.1.3); adversarial phase 3 |
| Long-session coherence (windowing loses global state) | Rolling context header carries buffer summary + progress; eval composition signatures at essay scale |
| Aalto/KLiCKe timestamp resolution or logging quirks differ | Normalize in the unified schema; per-corpus calibration checks before training |
| Dataset licenses restrict use | Both are research-released; verify license terms fit the intended use before training |

## 11. Build order (summary)

1. Data pipeline: download, parse, unify, label, serialize both corpora. **Milestone: round-trip
   replay of a real KLiCKe session from serialized tokens.**
2. Phase-1 motor fine-tune + transcription-mode eval (distributional + discriminator).
   **Milestone: Tier-1 realism achieved.**
3. Synthetic revision-trajectory generator.
4. Phase-2 composition fine-tune + constrained decoder.
5. Full eval suite; iterate (phase 3 if needed).
6. Serving deployment on Modal/Baseten.
