# The local portal

    uv run python scripts/playground.py                          # newest checkpoint
    uv run python scripts/playground.py --checkpoint checkpoints/motor-phase2
    uv run python scripts/playground.py --no-load                # pick one in the UI

Then open <http://localhost:8765>.

Bound to 127.0.0.1 with no host flag, deliberately: `motor-phase2` is
KLiCKe-derived, KLiCKe ships no license terms, and everything the model emits
inherits that hold. Do not expose it.

## What it is

`scripts/playground.py` is now only a CLI. The serving logic lives in
`typeshi.portal`, split so that each piece is testable without a model
resident — `registry` (checkpoint discovery and loading), `jobs` (the
one-at-a-time queue), `rows` (events to JSON), `readout` (per-sample realism)
and `corpus` (held-out sessions). `tests/test_portal.py` covers all of them
offline.

## Why it is asynchronous

Without the CUDA linear-attention kernels this architecture decodes at 13–17
tok/s on MPS, and the v2 grammar spends two tokens per keystroke. A sentence
is seconds; a paragraph is minutes. So:

- The port binds **before** the model loads. A cold start on a new base model
  pulls 9.3 GB, and the previous build-then-bind order left the browser
  staring at a dead localhost with no way to tell whether anything was
  happening. `/api/info` reports `status: idle|loading|ready|error`.
- Generation is a **job**, not a request. `POST /api/generate` returns a
  `job_id` immediately; `GET /api/jobs/{id}/stream` is server-sent events
  carrying token count, rate, ETA and the live buffer; `POST
  /api/jobs/{id}/cancel` stops it and still returns the partial stream.
- Exactly one job runs at a time — a correctness requirement, not only a
  memory one; the full rationale lives in `src/typeshi/portal/jobs.py`'s
  docstring, next to the code that enforces it.

Progress streaming and the typing animation are deliberately separate
channels. Generated timestamps are *simulated* milliseconds, not wall-clock,
so the animation replays the finished event list client-side — which is more
faithful than streaming and is what makes the speed control, replay and
scrub possible without re-running the model.

## The convergence guarantee is the default

There are two different guarantees in the codebase and only one of them binds
the output to your text:

- **Grammar mask** (`constrained=True`, the plain path): the output is a
  well-formed keystroke stream. Nothing checks it against the target, so a
  fumbled word is never corrected.
- **Convergence decoder** (`generate_windowed`): tracks the live buffer and
  masks the model's choices so it *cannot* finish on anything but the exact
  string. Typos still happen, then get backspaced and fixed.

The portal now defaults to the second, generating in the ~512-event windows
composition was trained on. Measured here on a 197-character paragraph,
`motor-phase2`, T=1.0:

| path | exact | events | backspaces |
| --- | --- | --- | --- |
| mask only, seed 2 | yes | 205 | 2.0% |
| mask only, seed 3 | yes | 205 | 2.0% |
| mask only, seed 4 | **no** (0.995) | 204 | 1.5% |
| convergence, seed 1 | yes | 495 | 30.1% |
| convergence, seed 2 | yes | 207 | 2.4% |
| convergence, seed 3 | yes | 725 | 36.4% |

Two things to take from that. The mask-only path silently drops a character
on roughly one seed in four — that is the "…domestic spher" failure, and it
is why the guarantee is on by default. But this table predates the staleness
and affordability fixes in `converge.py`, which turned out to be the cure
for the mask-fighting mode (re-measured after them: longest backspace run
~140 → 11; `docs/open-work.md`). A converged run that still lands high is
*correct* and *not realistic*, and the realism panel will say so; the
residual gap is revision training data, not the decoder.

Failures are named rather than counted: `ConvergenceError` (a `ValueError`,
so existing `except ValueError` callers keep working) carries the partial
stream, the per-window on-path progress and a `stalled` flag, and the portal
animates the partial run instead of discarding it.

## Draft → final revision

To watch the model revise rather than type, give it a draft. The portal's
"Start from a draft" field seeds the buffer (`generate_windowed(draft=...)`),
so convergence decoding has to produce the edit sequence that turns the draft
into the target. Measured, `motor-phase2`, draft "The revolution changed how
stuff was made." → target "The revolution reshaped how goods were made.":

| `staleness_window` | events | CUR | SELDEL | bksp | exact |
| --- | --- | --- | --- | --- | --- |
| 10 | 391 | 107 | 4 | 127 | yes |
| 120 | 232 | 23 | 9 | 54 | yes |

Both converge exactly. Note the looser window produces the *better* revision:
fewer events, more SELDELs, fewer caret hops. A tight window forces repair
before the model can commit to a bigger edit, so it thrashes the caret
instead of excising a phrase.

Two things this does not do. It will not make the model spontaneously write
a draft and then improve it — the mask can permit a long excursion but cannot
make that excursion a plausible earlier draft; unconstrained, this model
writes its own essay, so a long excursion is off-target text followed by a
bulk delete. That is the IteraTeR training gap. And the revision rate here
(14–28% of events) is far above real writers' 1.1–1.3%, because this is a
pure revision task rather than composition with occasional revision.

**`excursion_budget` is not the knob that limits detour length.** The gate is
`(depth < budget or affordable) and not resolving and not stale and not
floor` — four conjuncts, of which the budget is one disjunct of the first.
Since staleness scales with excursion depth (`STALE_DEPTH_SLACK`,
2026-08-24), deliberate excursions survive at the shipped defaults; raise
`staleness_window` only if a long revision still gets cut short.

## Two fixes this shook out

**EOS parity with the sampler.** The base config declares terminator 248044
(`<|endoftext|>`) while the fine-tuned tokenizer's EOS — the one the grammar
mask actually unmasks — is 248046 (`<|im_end|>`). So the model emitted its
EOS, `generate()` was watching for a different id and never stopped, and the
run burned its remaining budget on garbage that truncation then discarded.
`generate_session` now passes `terminator_ids(tok, model)` to
`model.generate` so the stop condition and the truncation read one set.
Identical output, much less compute: a 43-char transcription went 236 → 88
tokens (15.5 s → 6.5 s) and a composition run 592 → 112 tokens (35.2 s →
7.1 s). `run_eval` gets the same speedup for free.

**bf16 for inference.** `select_backend` pins fp32 on MPS, which is right for
a Trainer and wrong here — fp32 costs ~17 GB for the resized 4B where bf16
costs ~8.5, and since both the base shards and the adapter's embedding
tensors are stored bf16 on disk, bf16 is lossless. `portal.registry`
overrides only that, keeping `device_map=None`; the documented Apple-Silicon
segfault is bf16 *combined with* `device_map="auto"`, which this path never
forms. fp16 stays excluded (the gated-delta kernels carry an explicit
"A might be -inf" warning, and fp16 hangs on MPS at scale).

## What the realism panel will and will not claim

Per-sample, the panel shows the nine order-sensitive features (each with a
known null under exchangeability, so a gauge reading "null | this sample |
real-human band" needs no calibration) and refuses to show a per-sample KL,
a Fréchet distance, or a P(real) dial. Each refusal has a measured reason;
the full rationale with the numbers lives in the module docstring of
`src/typeshi/portal/readout.py`, next to the code that enforces it. Real
sessions are `codec_roundtrip`ed first; skipping that is how a discriminator
scores 0.915 on quantization alone.

Temperature ships with a warning rather than as a neutral slider: cooling
read monotonically harmful under the superseded pair-grouped eval
(1.0 → 0.60, 0.9 → 0.66, 0.8 → 0.685). The sweep has not been re-run
writer-grouped; T=1.0 stays the default.

## Checkpoint provenance

The picker lists every loadable save under `checkpoints/` — adapters and
full-weights alike — and reports what each was trained for, derived from the
corpus prefixes in its `split.json` (Aalto is transcription, KLiCKe is
composition). Selecting a mode the loaded checkpoint never trained on raises
a warning in the UI.

That matters because the two 4B checkpoints are not interchangeable and the
newer-looking one is not the later one:

| | `motor-full` | `motor-phase2` |
| --- | --- | --- |
| corpus | Aalto only, 132,559 writers | Aalto + KLiCKe, 5,792 |
| epochs / batch | 1 × 32 | 3 × 4, accum 8 |
| modes | transcription | transcription + composition |
| local held-out pool | 1,487 sessions | 23 |

`motor-phase2` was continued from `motor-full` via `--init-adapter`, so
Phase 1 is the *earlier* model despite arriving on this machine later. Its
split also matches far more of the local `heldout_aalto` logs, which makes
the corpus-comparison panel much richer under `motor-full`.

Composition on `motor-full` still converges — the mask guarantees that
regardless of training — but it thrashes noticeably more getting there
(measured on one sentence: 19.1% backspace events versus 12.5% for
`motor-phase2`). That is the mode mismatch, not a decoder fault.

## Known gaps

- Constrained composition now emits cursor ops and SELDELs (stage-2 ops plus
  the affordability rule; the draft-revision table above shows them live).
  What remains thin is spontaneous revision during ordinary composition:
  ~0.4% of events against real writers' 1.1–1.3%, a training-data gap
  (`docs/iterater-notes.md`), not a mask one. Unconstrained decoding can
  still produce out-of-buffer cursor ops; the portal renders the partial
  stream and names the defect instead of erroring.
- First load can be much slower than the ~25 s warm figure if the machine is
  under memory pressure (observed: 16 minutes while other large processes
  held RAM).
