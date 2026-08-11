# Tiny Motor Model PoC — Design

**Date:** 2026-08-10
**Status:** Approved direction (locked in brainstorming); revised after a
three-lens adversarial review; pending user review of this written spec.

## 1. Goal

A locally-trainable proof of concept for Phase 1: a **from-scratch ~19M-parameter
causal LM** that types — emitting the exact token-format-v2 event stream
(`<c:h>`/`<BKSP:h>`/`<DT:k>`, rollover by hold-bin > gap-bin comparison) for a
given transcription prompt — trained end to end on this Mac, evaluated by the
existing five-gate harness on held-out writers.

**What a hard-bar pass actually establishes** (calibrated deliberately — the
review caught the first draft overclaiming):

- the token format and dataset export are **learnable end to end** — a model
  trained only on `train.jsonl` copies targets through the full
  serialize→train→generate→deserialize loop;
- the constrained decoder and eval plumbing work against a second,
  independent model/tokenizer implementation;
- char-level target-copying is learnable at tiny scale.

**What it does *not* establish:** that the dataset's *timing* signal is sound
and learnable — that lives in the stretch gates. And the teeth/control gates
are computed from real sessions and the heuristic baseline only (the model's
generations never enter them); both already passed in the 0.8B overnight eval
(0.97 / 0.47), so here they serve as regression checks, not new evidence.

**Success bar** (per brainstorming):

- **Hard requirements:** `pass_generation_validity` (≥ 90% parse + type the
  target), `pass_discriminator_has_teeth`, `pass_control_near_chance`.
- **Stretch goal:** the two realism gates — `pass_model` (paired CV accuracy
  in [0.40, 0.55]) and `pass_serial_dependence_teeth`. These are where the
  timing data itself is newly tested. Reported as the headline result either
  way; missing them does **not** invalidate the PoC (capacity is a suspect at
  19M), but see §5a — a harness-ceiling control must run first so a realism
  failure can be attributed at all.

**Residual risks a PoC pass leaves open for the GPU run** (explicitly: a tiny
pass does *not* reduce the GPU run to "a pretrained 7B does it better"):

1. **BPE-target→char copying** — the 0.8B's dominant observed failure
   (103/120 wrong-text) is exactly the skill the char-level tokenizer
   engineers away. This stays the GPU pilot's top open risk; its first
   checkpoint gate should be wrong-text rate specifically.
2. Added-token registration over a 152k base vocab, embedding seeding,
   LoRA `modules_to_save`/`ensure_weight_tying`, bf16 CUDA numerics.
3. Realism at capacity, and knob generalization.

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

| knob | default (~19M) | smoke (~8M) |
|---|---|---|
| hidden size | 384 | 256 |
| layers | 8 | 6 |
| attention heads | 6 | 4 |
| **num_key_value_heads** | **6** | **4** |
| FFN intermediate | 1024 | 704 |
| vocab | 12,909 (see §3) | same |
| tie_word_embeddings | true | true |
| max_position_embeddings | 2048 | 2048 |

`num_key_value_heads` must be set explicitly — `Qwen2Config` defaults it to
32, which silently constructs a 29M-param model that crashes at the first
forward pass with 6 attention heads. Plain MHA (kv heads = heads) is what the
parameter counts above assume (verified: 19.13M / 8.13M, embeddings 4.96M of
the 19M, tied). Full-parameter training from random init (standard config
init, std 0.02). The model's `generation_config` sets
`eos_token_id`/`pad_token_id` to the custom tokenizer's IDs.

## 3. Custom small tokenizer

The one genuinely new component: `src/typeshi/tiny_tokenizer.py`, building a
`PreTrainedTokenizerFast` programmatically (no vocab files to maintain):

- **Char-level WordLevel model** over the 97 text characters: printable ASCII
  0x20–0x7E plus `\n` and `\t`.
- **All 12,810 grammar/knob/marker tokens** registered as added tokens —
  single IDs, mirroring `prepare_tokenizer()` on Qwen. Prefix-only entries
  (`<CUR:`, `<SELDEL:`) are *not* registered, exactly as there; transcription
  never emits them.
- `<EOS>` and `<PAD>` specials. **No `<UNK>`** — total 12,909. The WordLevel
  model is constructed *without* an `unk_token`, which is precisely what makes
  encoding raise on out-of-vocab input (wiring an unk token would silently
  swallow OOV chars and violate requirement 3 below).

Char-level target text is a feature for *this model* — copying becomes a
monotonic char→key attention pattern, the easiest skill small transformers
learn — but note it is also why a PoC pass does not discharge the GPU run's
BPE-copying risk (§1, residual risk 1).

**Hard requirements, each pinned by a test before anything trains:**

1. Every registered grammar token encodes to exactly **one ID** (the
   `prepare_mlx_model.py` verification, reused).
2. **Byte-exact decode**, no separator insertion: `generate()` decodes with
   `skip_special_tokens=False` and hands the string to `deserialize`, which
   rejects stray whitespace by design. The tokenizer's decoder must fuse
   tokens with no joiner.
3. Encoding text containing an out-of-vocab character **raises** rather than
   silently mapping anywhere.
4. All of the above hold **after a `save_pretrained` →
   `AutoTokenizer.from_pretrained` round-trip**, not just on the in-memory
   object — added-token serialization is exactly where decode behavior can
   shift, and a drift here would surface only after an overnight run as 100%
   "malformed" generations.

**OOV containment** (the build does not guarantee prompt-side coverage:
`unsupported_chars()` gates *typed* chars only, never the target sentence):

- `train_tiny.py` pre-scans prompts and drops (and counts) examples whose
  prompt fails to encode;
- the eval skips-and-counts a session whose prompt fails to encode instead of
  crashing (the tokenizers OOV error is not the `ValueError` the loop already
  catches).

## 4. Data flow and integration seams

Unchanged and consumed as-is:

- `scripts/build_dataset.py` output: `train.jsonl` (prompt/completion) and
  `split.json` (writer holdout).
- `build_prompt` / `serialize` / `deserialize` / `timebins`.
- `generate()` in `generate.py` — model-agnostic already.
- `GumbelSampleProcessor`, the discriminator, distributional metrics, gates.

New: `src/typeshi/train_tiny.py` — argparse entry point mirroring
`train_motor.py`: same SFTTrainer prompt/completion recipe (loss masked to the
completion), same `<MODE:T>` filter, same `split.json` copy into the
checkpoint, reusing `select_backend()` (fp32 on MPS). Differences: builds the
tiny config + tokenizer, no PEFT, no embedding seeding, saves with
`save_pretrained`, adds a **`--limit N` flag — a seeded random sample taken
after the mode filter** (random, not head-of-file: `train.jsonl` ordering
groups writers, and the pilot's data-scaling read needs full writer breadth),
plus the OOV pre-scan (§3) and checkpointing/resume settings (§5).

Three small edits to existing code (the first draft claimed one; the review
found two more were required):

1. `scripts/run_eval.py`: load `AutoPeftModelForCausalLM` only when the
   checkpoint contains `adapter_config.json`, else plain
   `AutoModelForCausalLM`.
2. `scripts/run_eval.py`: skip-and-count sessions whose prompt fails to
   encode (§3 OOV containment).
3. `src/typeshi/constrain.py`: **legalize EOS in the gap slot** (once the
   stream is non-empty). TRL appends EOS directly after the completion's
   final event token — a *gap* position under the alternating grammar — so
   every model is trained to emit EOS exactly where the current mask forbids
   it. A pretrained model sometimes terminates anyway; a from-scratch model
   has no prior to fall back on and would burn its full budget, failing
   validity for a mechanical reason that would masquerade as a capacity
   failure. **This fix applies to Phase 1's GPU run too** (Qwen's appended
   `<|im_end|>` lands in the same slot) and may explain part of the 0.8B's
   overtyping; it goes in regardless of the PoC's fate. A stream ending
   EOS-after-event stays clean — no dangling `<DT:>`.

## 5. Training and eval plan

Staged, each stage a kill-switch for the next:

1. **Fixture e2e** — smoke config, ~10 steps on the test fixtures: loss
   falls, checkpoint saves, and the trained checkpoint is **reloaded from
   disk exactly as `run_eval.py` loads it** (AutoTokenizer + the new plain
   loader branch), then constrained generation parses and emits EOS before
   budget.
2. **Throughput smoke** — smoke config, `--limit 1000` real examples.
   Measures: (a) sustained **examples/sec** (the honest unit — tokens/sec
   is distorted by the char-level prompt inflation), and (b) the time and
   memory of the HF datasets tokenize-map, extrapolated ×2,000 to the full
   corpus (pre-tokenize to disk if the extrapolation is unacceptable).
   Go/no-go ladder for stage 4, from measured ex/s: **≥ 46 ex/s** → full
   epoch fits overnight (≤ 12 h); **23–46** → half-corpus night, second
   night resumes the rest; **< 23** → stop and rethink (MLX port of the tiny
   model, or accept multi-night). For scale: 1.99M examples ≈ **~280M
   tiny-tokenizer tokens** (91.4 completion + ~47 char-level prompt ≈ 138
   tokens/example; the earlier 460M figure was wrong and is retracted).
3. **Pilot** — 19M config, **two runs: `--limit 25000` and `--limit
   100000`**, 1 epoch each, same seed, each followed by `run_eval.py --n 50`
   (attempt cap 150 → validity has roughly ±6-point uncertainty; bands below
   respect that). Decision bands:
   - **proceed** if 100k-validity ≥ 40%, or ≥ 2× the 25k-validity;
   - **stop** if 100k-validity < 5% after one predeclared lever (a second
     epoch on the 100k subset);
   - **between:** apply the same lever once, then force a re-decision with
     no further levers (next stop: the encoder-decoder fallback below).
4. **Full run** — one epoch over all 1.99M transcription examples per the
   stage-2 ladder, under `caffeinate`, with `save_steps` set to ~30–60 min
   of measured throughput, `save_total_limit=2`, and
   `resume_from_checkpoint` on relaunch — a crash at hour 7 must cost one
   save interval, not the night. Then `run_eval.py --n 200` on held-out
   writers, same gates, same report JSON.

Hyperparameters (starting points, tuned only if the pilot demands):
AdamW, lr 6e-4 with warmup_ratio 0.03 and cosine decay (a fixed 500-step
warmup would consume the entire ~390-step 25k pilot), effective batch 64
(per-device 16 × accum 4; MPS dispatch overhead dominates small models, so
per-device size is worth revisiting at the throughput smoke), weight decay
0.1, grad clip 1.0, seed from `config.DEFAULT_SEED`. fp32 on MPS per
`select_backend()`.

**Encoder-decoder fallback** (named by the pilot kill-switch): a char-level
encoder (~4 layers) over target + knobs, an event-token decoder (~6 layers)
with cross-attention. The tiny tokenizer, serialize/deserialize, gates, and
report format survive unchanged; `generate()`'s causal-LM path and the
grammar processor need a seq2seq generation variant. Only sketched here —
it gets its own design pass if ever triggered.

### 5a. Harness-ceiling control (before the full run)

The realism gates compare fakes that exit `deserialize()` carrying
bin-center timings against real sessions carrying raw milliseconds. Whether
that asymmetry alone is discriminable has **never been measured** (the
documented 0.085-on-exact-copies calibration used raw copies, not
round-tripped ones). One-off control, run before the overnight spend:
paired discriminator accuracy of real vs `serialize→deserialize`(real) —
same sessions round-tripped through the token format — full-feature and
timing-only variants. If it lands outside ~[0.40, 0.55], `pass_model` is
unreachable for *any* generator including the 7B, and the featurization
(quantize real timings through the bins, or jitter fakes within-bin) must be
fixed before any realism number — PoC or GPU — is interpreted. Either way,
record the measured ceiling in the results doc; it also recalibrates the
0.8B's 0.77.

## 6. Testing

Matching the repo's existing test style (`tests/test_train_motor.py` pins
`select_backend`; MLX prep verifies single-token survival):

- **Tokenizer property tests** — run against both the in-memory tokenizer
  **and its save/load round-trip copy** (§3 req. 4): grammar tokens
  single-ID; byte-exact round-trip of a real fixture prompt+completion pair;
  OOV char raises; `<EOS>`/`<PAD>` IDs stable and distinct.
- **Constrain-compat tests:** `TranscriptionGrammarProcessor` built from the
  tiny tokenizer yields non-empty event/DT ID sets and masks a text char's
  ID; EOS is legal in the gap slot once the stream is non-empty and illegal
  at position zero (pinning the §4.3 fix, for the extended-Qwen tokenizer
  path too).
- **Eval-loader test:** checkpoint dirs with and without
  `adapter_config.json` route to the right loader class.
- **End-to-end micro-test:** stage-1 of §5 as a slow-marked test — train the
  smoke config ~10 steps on fixtures, reload from disk the way the eval
  does, generate constrained, assert EOS arrives before budget and
  `deserialize` succeeds.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| MPS fp32 throughput too low for an overnight epoch | Throughput smoke measures ex/s first; explicit ladder (full / half+resume / stop) instead of hope |
| Full-corpus tokenize-map blows time or RAM before step 1 | Measured and extrapolated at the throughput smoke; pre-tokenize to disk if needed |
| Overnight run dies mid-flight with nothing to resume | `save_steps` ≈ 30–60 min, `save_total_limit=2`, `resume_from_checkpoint`, `caffeinate` |
| From-scratch model can't terminate under the mask (EOS trained in a slot the mask forbids) | Fixed by design in §4.3; pinned by the constrain-compat and e2e tests |
| Prompt contains a char the tokenizer can't encode (build never gated prompt text) | Pre-scan drops-and-counts in training; eval skips-and-counts (§3) |
| Tokenizer save/load drifts and the eval rejects everything after the overnight spend | §3 req. 4: all property tests run on the round-tripped tokenizer; e2e test reloads from disk |
| 19M can't reach 90% validity (copying fails) | Two-point pilot with numeric bands catches it early; encoder-decoder variant is the named fallback before any GPU spend |
| Realism gates fail — on capacity, or on a harness artifact | §5a control separates the two before the number is interpreted; realism is out of the hard bar either way |
| A tiny-model pass overstates what it proves | §1 scopes the claim: format/plumbing/copy-learnability proven; timing soundness only via stretch gates; GPU residual risks listed explicitly |
| TRL/SFTTrainer quirks with a from-scratch model | The prompt/completion path already avoids chat templates; fixture e2e flushes the rest at minutes-scale cost |

## 8. Build order

1. `tiny_tokenizer.py` + property tests (incl. save/load round-trip).
   **Milestone: byte-exact round-trip of real fixture examples, both copies.**
2. `constrain.py` EOS-in-gap fix + constrain-compat tests (also benefits
   Phase 1 regardless of PoC outcome).
3. `train_tiny.py` (incl. `--limit`, OOV pre-scan, checkpointing) + fixture
   e2e micro-test. **Milestone: loss falls; e2e test passes against the
   disk-loaded checkpoint.**
4. `run_eval.py` edits (loader fallback, OOV skip) + loader test.
5. Harness-ceiling control (§5a). **Milestone: measured
   real-vs-roundtripped ceiling recorded.**
6. Throughput smoke (`--limit 1000`). **Milestone: measured ex/s + map
   extrapolation → ladder decision.**
7. Two-point pilot (25k / 100k) + evals. **Milestone: a decision band hit,
   with both validity numbers recorded.**
8. Full run per ladder + `run_eval.py --n 200`. **Milestone: hard gates
   pass; realism gates + §5a ceiling reported together.**
9. Results doc alongside `results-08b-shakedown.md`.
