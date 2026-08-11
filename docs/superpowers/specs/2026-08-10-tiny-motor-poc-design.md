# Tiny Motor Model PoC — Design

**Date:** 2026-08-10
**Status:** Approved direction (locked in brainstorming); pending user review of this written spec.

## 1. Goal

A locally-trainable proof of concept for Phase 1: a **from-scratch ~19M-parameter
causal LM** that types — emitting the exact token-format-v2 event stream
(`<c:h>`/`<BKSP:h>`/`<DT:k>`, rollover by hold-bin > gap-bin comparison) for a
given transcription prompt — trained end to end on this Mac, evaluated by the
existing five-gate harness on held-out writers.

**What it proves:** the dataset, token format, constrained decoder, and eval
harness are sound. If a 19M model trained overnight clears the bar, the only
claim the rented-GPU run still has to prove is "a pretrained 7B does it better."

**Success bar** (per brainstorming):

- **Hard requirements:** `pass_generation_validity` (≥ 90% parse + type the
  target), `pass_discriminator_has_teeth`, `pass_control_near_chance`.
- **Stretch goal:** the two realism gates — `pass_model` (paired CV accuracy in
  [0.40, 0.55]) and `pass_serial_dependence_teeth`. Reported as the headline
  result either way; missing them does **not** invalidate the PoC, because
  capacity, not data, is the suspect at 19M.

### Non-goals

- Composition mode, `<CUR:>`/`<SELDEL:>`, revision behavior of any kind.
- Replacing Phase 1: this is a de-risking step, not a product model.
- Pretraining, PEFT/LoRA, GPU rental, serving.
- Filtering the data: it trains on the same transcription export Phase 1 uses,
  typos and backspaces included, ECOR/EUNC knobs intact.

## 2. Architecture

A randomly-initialized **stock `Qwen2ForCausalLM`** with a small config — no
custom modeling code, so the checkpoint is a plain HF model directory that
`model.generate()`, `TranscriptionGrammarProcessor`, and the eval understand
natively.

| knob | default (~19M) | smoke (~9M) |
|---|---|---|
| hidden size | 384 | 256 |
| layers | 8 | 6 |
| attention heads | 6 | 4 |
| FFN intermediate | 1024 | 704 |
| vocab | ~12,910 (see §3) | same |
| tie_word_embeddings | true | true |
| max_position_embeddings | 2048 | 2048 |

Tied embeddings are ~5M of the 19M. Full-parameter training from random init
(standard config init, std 0.02). The model's `generation_config` sets
`eos_token_id`/`pad_token_id` to the custom tokenizer's IDs so `generate()`
terminates correctly.

## 3. Custom small tokenizer

The one genuinely new component: `src/typeshi/tiny_tokenizer.py`, building a
`PreTrainedTokenizerFast` programmatically (no vocab files to maintain):

- **Char-level WordLevel model** over the 97 text characters that survive the
  dataset build: printable ASCII 0x20–0x7E plus `\n` and `\t`. These cover
  every `<TARGET>` text (`unsupported_chars()` already drops the rest).
- **All 12,810 grammar/knob/marker tokens** registered as added tokens —
  single IDs, mirroring `prepare_tokenizer()` on Qwen. Prefix-only entries
  (`<CUR:`, `<SELDEL:`) are *not* registered, exactly as there; transcription
  never emits them.
- `<EOS>`, `<PAD>`, `<UNK>` specials. Total ≈ 12,910.

Char-level target text is a feature, not a compromise: copying becomes a
monotonic char→key attention pattern — the easiest skill small transformers
learn — instead of BPE-piece-to-char untangling.

**Hard requirements, each pinned by a test before anything trains:**

1. Every registered grammar token encodes to exactly **one ID** (the
   `prepare_mlx_model.py` verification, reused).
2. **Byte-exact decode**, no separator insertion: `generate()` decodes with
   `skip_special_tokens=False` and hands the string to `deserialize`, which
   rejects stray whitespace by design. The tokenizer's decoder must fuse
   tokens with no joiner.
3. Encoding text containing an out-of-vocab character **raises** rather than
   mapping to `<UNK>`. The dataset build guarantees none exist; silence here
   would hide a regression.

## 4. Data flow and integration seams

Unchanged and consumed as-is:

- `scripts/build_dataset.py` output: `train.jsonl` (prompt/completion) and
  `split.json` (writer holdout).
- `build_prompt` / `serialize` / `deserialize` / `timebins`.
- `generate()` in `generate.py` — model-agnostic already.
- `TranscriptionGrammarProcessor` and `GumbelSampleProcessor` — both build
  their state from the tokenizer generically.
- The discriminator, distributional metrics, and all five gates.

New: `src/typeshi/train_tiny.py` — argparse entry point mirroring
`train_motor.py`: same SFTTrainer prompt/completion recipe (loss masked to the
completion), same `<MODE:T>` filter, same `split.json` copy into the
checkpoint, reusing `select_backend()` (fp32 on MPS). It differs only in
what it trains: builds the tiny config + tokenizer, no PEFT, no embedding
seeding (nothing pretrained to preserve), saves with `save_pretrained`.

Changed (the only edit to existing code): `scripts/run_eval.py` loads
`AutoPeftModelForCausalLM` only when the checkpoint contains
`adapter_config.json`, else plain `AutoModelForCausalLM` (~5 lines). Everything
downstream of loading is untouched.

## 5. Training and eval plan

Staged, each stage a kill-switch for the next:

1. **Smoke** — 9M config, ~1k examples, minutes on MPS. Confirms: loss falls,
   checkpoint saves and reloads, constrained generation parses. **Measures
   real MPS throughput** — the 8–10 h/epoch estimate below is back-of-envelope
   until this number exists.
2. **Pilot** — 19M config, ~100k examples (`--limit`-style subset), then
   `run_eval.py --n 50`. Decision point: validity clearly above the 0.8B's
   14% and trending with data → proceed. Validity flat near zero → stop and
   revisit (first lever: more epochs on the pilot subset; second: the
   encoder-decoder fallback from brainstorming).
3. **Full run** — one epoch over all 1.99M transcription examples, overnight
   (~460M tokens; estimated 8–10 h at fp32 on the M5 Pro, to be replaced by
   the smoke measurement). Then `run_eval.py --n 200` on held-out writers,
   same gates, same report JSON.

Hyperparameters (starting points, tuned only if the pilot demands):
AdamW, lr 6e-4 with 500-step warmup and cosine decay, effective batch 64
(per-device 16 × accum 4), weight decay 0.1, grad clip 1.0, seed from
`config.DEFAULT_SEED`. fp32 on MPS per `select_backend()`.

## 6. Testing

Matching the repo's existing test style (`tests/test_train_motor.py` pins
`select_backend`; MLX prep verifies single-token survival):

- **Tokenizer property tests:** grammar tokens single-ID; byte-exact
  round-trip of a real prompt+completion pair from the fixtures; OOV char
  raises; `<EOS>`/`<PAD>` IDs stable and distinct.
- **Constrain-compat test:** `TranscriptionGrammarProcessor` built from the
  tiny tokenizer yields non-empty event/DT ID sets and masks a text char's ID.
- **Eval-loader test:** checkpoint dirs with and without
  `adapter_config.json` route to the right loader class.
- **End-to-end micro-test:** train the smoke config ~10 steps on the test
  fixtures, generate constrained, `deserialize` succeeds and replays onto
  the target's first characters. Slow-marked, like the existing e2e tests.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MPS fp32 throughput too low for an overnight epoch | Smoke stage measures it first; fall back to a half-corpus epoch — the pilot's data-scaling read says whether that suffices |
| 19M can't reach 90% validity (copying fails) | Pilot decision point catches it early; encoder-decoder variant is the named fallback before any GPU spend |
| Realism gates fail on capacity | Explicitly out of the hard bar; report numbers and the validity/teeth result stands |
| Tokenizer decode inserts separators and everything "is malformed" | Hard requirement §3.2 with a test before training; `deserialize` rejecting stray whitespace makes this loud, not silent |
| TRL/SFTTrainer quirks with a from-scratch model (no chat template, custom tokenizer) | The prompt/completion path already avoids chat templates in `train_motor.py`; the smoke stage exists to flush the rest out at minutes-scale cost |
| A tiny-model pass overstates what it proves | The spec's claim is calibrated: data/format/harness validated; LLM-specific claims (composition, knob generalization) explicitly stay open |

## 8. Build order

1. `tiny_tokenizer.py` + property tests. **Milestone: byte-exact round-trip
   of real fixture examples.**
2. `train_tiny.py` + smoke run on fixtures. **Milestone: loss falls; e2e
   micro-test passes.**
3. `run_eval.py` loader fallback + test.
4. Pilot run + eval. **Milestone: validity materially above the 0.8B's 14%.**
5. Full overnight run + `run_eval.py --n 200`. **Milestone: hard gates pass;
   realism gates reported.**
6. Results doc alongside `results-08b-shakedown.md`.
