# Typeshi

Typeshi trains language models to type like people. The output is not text but the keystroke stream that produces it: every press with its hold time, the gap to the next key, typos and the backspaces that chase them, the long pause before a clause, the cursor jumping back into a draft to reword a sentence. Give a model a target text and a typist profile (speed, error rates, revision rate) and it types the text, either in transcription mode (copy this sentence, the Aalto task) or composition mode (write this essay, with the pauses and revisions that implies).

The models are judged adversarially. A classifier is first proven able to catch naive fakes and shuffled timings, then asked to tell generated sessions from real held-out human typing. The project passes when it can't.

This is non-commercial research. Keystroke data is usually studied to identify typists; Typeshi runs the other direction and builds models that reproduce typing behavior rather than recognize the person behind it.

## What the typing process can teach

A keystroke log records what the finished document throws away, and several of the things it records are interesting on their own.

Where writers pause is where the thinking surfaces. Real writers pause at clause boundaries rather than mid-word, and the composition eval scores generated typing on exactly that: inter-key intervals classified as clause boundary, word boundary, or within word, read from the live text buffer, plus P-burst lengths (the writing-research convention for runs of typing between pauses over two seconds). Getting these right means the model has learned something about where planning happens, not just how fast fingers move.

How people draft and revise only exists in the process. The final text keeps no record of the sentence that got typed, reconsidered, and replaced. The composition eval measures revision behavior directly (backspace fraction, cursor moves, range deletions, episode lengths), and it is exactly where models currently fall short: the discriminator wins on revision statistics, not on timing. One measured finding in `docs/` is that the model was willing to revise all along and the decoding mask was what forbade it; the remaining gap is training data, since deliberate revision is nearly absent from the corpus. The IteraTeR dataset (human-annotated revision chains: 4,018 labelled edits with character offsets and intent labels like clarity and fluency) is the bridge to richer revision behavior, and its adapter is already wired in behind a build flag.

Timing alone may encode the physical keyboard. The `interp/` probes take a 27x27 matrix of digraph latencies and try to rebuild the two-dimensional key layout from it blind: no labels, no layout prior, scored afterward against real ANSI QWERTY geometry. The pipeline is validated on synthetic matrices with planted confounds; reading the trained model with it is pending. The How We Type corpus, where motion capture labels the finger behind every keypress, gives the same-finger rule supervised ground truth. The docs call this the keyboard-reconstruction workstream.

Behavior can be separated from hardware. Two requested corpora extend the reach here: Clarkson II is uninstructed and longitudinal (how typing varies across months and applications), and Buffalo CUBS has 73 of its 148 subjects typing on a different keyboard each session, which is exactly what separating typist from hardware requires.

And a model that types like a person is an instrument in itself: realistic input for HCI experiments, plausible earlier drafts for revision research, or simply the portal demo, which types any sentence you give it on an animated keyboard, key rollover included.

## How it works

Raw keystroke corpora become canonical event streams, event streams become tokens, a fine-tuned LLM learns to emit those tokens, a constrained decoder guarantees the output types the right text, and an adversarial eval decides whether the result looks human. The design decisions below all carry their measurements, and most files in `docs/` are experiment write-ups with the numbers inline.

### Data

Three keystroke corpora are wired in. Aalto 136M Keystrokes is the transcription corpus: 165,008 physical-keyboard files of participants copying short sentences (around 43 characters each), yielding 1.95M training examples. KLiCKe is the composition corpus: 4,992 essay-writing sessions logged keystroke by keystroke, with cursor movement and range deletions. How We Type is smaller but carries a motion-capture finger label on every keypress; it feeds the keyboard-reconstruction workstream rather than training. A fourth adapter ingests IteraTeR, the revision dataset, behind an opt-in flag on `build_dataset.py`.

Each corpus has an adapter (`src/typeshi/adapters/`) that parses its native log format into one canonical event stream with four event types: KEY, BACKSPACE, CURSOR, SELDEL. The adapters were written against the actual bytes, not the papers; `docs/data-schemas.md` records where each corpus's own documentation disagrees with its files.

Integrity comes from a replay gate: parsed events are replayed through a text buffer and the result compared to the corpus's own ground truth (edit similarity of at least 0.90 for Aalto and How We Type, byte-exact for KLiCKe). Sessions that fail are dropped, never patched. So are sessions containing paste events: in one 500-log sample, 142 paste rows expanded into 15,031 zero-interval "keystrokes", and training on that teaches instant multi-character typing, the exact failure mode the model exists to avoid.

Every session gets four behavioral labels: words per minute, corrected error rate, uncorrected error rate, and revision rate. These become the prompt knobs, so the model learns the knob-to-behavior mapping from real variation between typists. Speed, correction rate, and revision rate are recomputed per 512-event training window (uncorrected error rate stays session-wide by design), because the session-level revision label matched its own window only 52.9% of the time. The train/test split holds out writers, never sessions, by hashing each writer id, so a subset build and the full build agree on every writer's assignment.

### The token format

One keystroke is two tokens. `<e:51>` means the letter e held for time bin 51; the `<DT:48>` after it is the gap to the next press. Holds and gaps share a single 128-bin log-spaced scale from 1 ms to 120 s, which makes key rollover a plain integer comparison: in `<X:y><DT:z>`, y greater than z means X was still down when the next key went down. 26% of real keystrokes overlap this way, so it has to be representable.

The grammar is 12,810 registered tokens: 97 character identities (92 printable ASCII plus SPC, NL, TAB, LT, GT) times 128 hold bins, 128 `<DT:>` bins, 128 `<BKSP:>` bins, and the prompt vocabulary. Cursor moves and range deletions (`<CUR:p>`, `<SELDEL:a-b>`) stay plain text parsed by regex, since they carry unbounded integers and account for under 1% of events.

A training example is a prompt and a completion. The prompt is five single-token knobs, the target text in natural language, then a process marker:

```
<MODE:C><WPM:14><ECOR:3><EUNC:0><REV:5><TARGET>the essay text...<PROCESS>
```

The knob values are bin indices, not literal units. The target staying natural language is the argument for using an LLM at all: pretraining earns its keep in composition, where a plausible essay has to come from somewhere. Loss is computed on the completion only; version 1 of the format spent about 27% of its training signal predicting its own prompt. Long sessions split into windows of at most 512 events, and a continuation window's prompt carries the last 500 characters of buffer plus the caret position, with the real gap across the boundary (in composition, often a minutes-long think) opening the completion. The full spec, with the measurement behind every choice, is `docs/token-format.md`.

### Training

`python -m typeshi.train_motor` LoRA fine-tunes a pretrained base model (currently Qwen3.5-4B; Qwen2.5-7B-Instruct was the previous default and remains supported). The tokenizer grows by 12,810 tokens, and each new embedding is seeded from the mean of the sub-word pieces that token used to shatter into, so `<DT:50>` starts near `<DT:51>` and `<e:51>` starts near the letter e. The embedding table trains alongside the adapters by default, since attention-only LoRA never touches it.

`python -m typeshi.train_tiny` trains a 19M-parameter model from scratch over a closed char-level vocabulary. It exists as a proof of concept and delivered one: trained overnight on a Mac (7h50m on an M5 Pro), it typed held-out targets with 100% validity, against 14.2% for a pretrained 0.8B baseline that saw a hundredth of the data. Target-copying turns out to be a data-volume skill.

There is also an MLX route for training on Apple Silicon (`scripts/prepare_mlx_model.py`, with measurements in `docs/training-on-apple-silicon.md`); the production path is a rented CUDA box provisioned by `scripts/setup_gpu.sh`. `docs/gpu-run-chronicle.md` is the complete, self-audited record of one such rental, incidents included.

### Generation

Free-running generation does not work. KLiCKe taught the model "write like this", and nothing at sampling time binds it to your target: probed, it started the right essay only 2 times in 5. The guarantee therefore comes from the mask, not from training.

`ConvergenceProcessor` (`src/typeshi/converge.py`) maintains a live text buffer during decoding and masks logits so the buffer provably converges to the target. Within that constraint the model keeps its freedom: the next needed key is always available, typos are allowed while an excursion budget lasts, and every timing token stays fully model-sampled, which is what keeps the realism numbers meaningful. Deliberate revisions run through an affordability check: an off-target excursion of any length is allowed while the remaining token budget can still pay for the cheapest route back that the mask can compel (cursor move, one range delete, retype), so the model revises early and polishes late. A staleness counter separates a typo the model typed past (fix it now) from a revision in progress (give it room). A generation that never converges raises `ConvergenceError` carrying the partial events and a named reason; it is a counted failure, never a silently wrong text.

Sampling is Gumbel-argmax rather than `torch.multinomial`, because the multinomial kernel on Apple's MPS backend was caught emitting tokens whose logits were verifiably minus infinity, roughly once per 250 steps.

### Evaluation

The pass condition is that a discriminator cannot tell. `scripts/run_eval.py` (Tier 1, transcription) trains a gradient-boosted classifier on per-session features (timing quantiles, event counts, and nine serial-order statistics) and requires five gates at once: the discriminator must catch a naive heuristic baseline at 0.90 or better and timing-shuffled real sessions at 0.75 or better, a real-vs-real control must land near chance, at least 90% of generations must parse and type their target, and real-vs-generated accuracy must land in [0.40, 0.55]. The band's lower edge matters: below-chance accuracy means leakage, not realism.

Real sessions are projected through the codec (serialize, then deserialize) before scoring, because generated timings are born on the 128-bin grid and raw ones are not. Without this, quantization alone separates real from generated at 0.915 and no token-emitting model can pass. Cross-validation folds are grouped by writer, because pair-grouped folds let the classifier use typist identity as a proxy for realness (measured: 0.63 against 0.52).

Tier 2 (`scripts/run_eval_composition.py`) adds convergence rate, pause-placement signatures, revision statistics, and knob fidelity (requested versus realized, as correlation and error). Signature and knob numbers ship without pass thresholds for now, on the stated principle that a gate invented before its distribution is known is a gate tuned to pass.

Every experiment's report is committed at the repo root (the `eval_report*.json` and `eval_tiny*.json` files), and each gate exists because a review found the eval gameable without it. Three of the project's worst bugs were in the harness, not the models: an inverted end-of-stream rule in the constrained decoder that made a 99.5%-valid model read as 1% valid, the quantization comb above, and a tokenizer fallback that silently ate every space.

## Results so far

Tier-1 is met. Under the corrected evaluation protocol (writer-grouped folds, 200 sessions spread across 67 writers instead of 14), `motor-full` scores 0.5129 against real typing and `motor-phase2` scores 0.5100, with generation validity 1.000 and 0.995: all five gates pass for both checkpoints.

The path there is the instructive part. Every earlier failure was the same artifact: pair-grouped folds let the discriminator recognize the typist rather than judge realism, since real sessions carry a writer's fingerprint and generated ones cannot. Under that leak the gate read 0.77 on the 0.8B shakedown, 0.640 on the tiny model, and a stubborn 0.58 to 0.60 plateau on the 4B, and four levers were spent chasing it: 8x more data moved the marginal distributions but not the gate, a third epoch did nothing, lowering temperature was monotonically harmful (cooling pulls generation variance below human variance, and the discriminator reads exactly that), and RAFT round 1 came back null and is reported as null. One pre-fix observation still worth keeping: a tiny-model sibling trained on a twentieth of the data, typing only 77% of targets correctly, scored closer to human than its fully converged sibling, because a converged model sharpens toward its mean and the mean is more regular than any individual human.

Composition (checkpoint `motor-phase2`) is further behind but moving. Windowed generation, matching the 512-event training horizon, lifted convergence from 76.3% to 90.1% over held-out KLiCKe essays with zero malformed streams. Timing is close to solved: hold-time KL 0.008, pause KL 0.026, pause-position classes at 0.13 to 0.17. The docs' summary is that the model pauses like a writer. The discriminator still wins on revision behavior (0.945 in the windowed run, a pair-grouped figure to be read as an upper bound), and the diagnosis moved from "the model can't revise" through "the mask was built for transcription typos and forbade revision" (fixed: the longest backspace run fell from about 140 to 11) to a training-data problem: 87% of composition windows sit at a revision rate of 1% or below, so the behavior being asked for is nearly absent from the data. That is the current frontier, and the IteraTeR synthesis path exists for it. `docs/open-work.md` keeps the ranked list of what comes next.

## The portal

`uv run python scripts/playground.py` starts a local web playground at http://127.0.0.1:8765. Type a sentence, pick a mode, set the knobs, and watch the model type it in real time on a simulated keyboard: each key lit for exactly its hold duration, rollover included, corrections and revisions as they happen. It can replay a real held-out human session next to the model's on the same gauges, seed the buffer with a draft to watch pure revision behavior, scrub finished sessions at 0.25x to 4x, and hot-swap checkpoints. The realism panel shows nine serial-order features against the band of real human sessions, with two reference points: a naive simulator the eval reliably catches, and the same sample with its timing shuffled.

The readout is deliberately narrower than the eval report. Per-sample KL against a pooled reference would mark genuine humans inhuman (a real person scores 3.59 where the pooled figure is 0.014), and no calibrated probability of realness exists to show, so the panel refuses to display either. `docs/portal.md` covers the details.

The server binds 127.0.0.1 and that is not configurable: the phase-2 checkpoint is KLiCKe-derived and KLiCKe ships no license terms, so nothing it generates leaves the machine.

## Repository layout

```
src/typeshi/
  adapters/        corpus parsers: aalto, howwetype, klicke, iterater
  events.py        the canonical event stream (KEY, BACKSPACE, CURSOR, SELDEL)
  serialize.py     token format v2: events <-> tokens, codec_roundtrip
  timebins.py      the shared 128-bin log time scale
  labels.py        per-session and per-window behavior labels
  dataset.py       windowing, prompt building, revision oversampling
  corpus_build.py  parallel corpus sweep (byte-identical at any worker count)
  train_motor.py   LoRA fine-tune of the base model
  train_tiny.py    19M from-scratch proof of concept
  constrain.py     transcription grammar mask
  converge.py      convergence-guaranteed decoding
  generate.py      sampling harness, windowed generation
  buffer.py        text-buffer replay, the single source of truth
  eval/            discriminator, distributional metrics, signatures, knobs
  interp/          keyboard-reconstruction probes
  portal/          the local playground's server
scripts/           build_dataset, run_eval, run_eval_composition, gen_raft_data,
                   playground, watch_training, setup_gpu.sh, ...
docs/              design notes, experiment write-ups, runbooks, data schemas
tests/             the test suite, offline by default
eval_*.json        committed evidence for the numbers cited in docs/
```

## Getting started

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Core dependencies are polars, numpy, scipy, and scikit-learn; the training stack (torch, transformers, peft, trl) is behind the `train` extra.

```
uv sync --extra train --extra dev
uv run pytest        # offline by default; network and slow tests are env-gated
```

The corpora are not in the repo (`data/` is gitignored) and must be obtained under their own terms: Aalto from the 136M Keystrokes project page, KLiCKe and How We Type from their publishers. They land under `data/raw/` (`data/raw/aalto`, `data/raw/klicke`), which is where `build_dataset.py` looks by default. Then:

```
uv run python scripts/build_dataset.py     # data/processed/{train,test}.jsonl + split.json
uv run python -m typeshi.train_tiny        # or -m typeshi.train_motor for the full fine-tune
uv run python scripts/run_eval.py --checkpoint checkpoints/motor-tiny
uv run python scripts/playground.py
```

## Licensing and data use

The code in this repository currently carries no license file.

The data is the binding constraint:

| Source | Terms | What that means here |
|---|---|---|
| Aalto 136M Keystrokes | non-commercial research | transcription training; the project's non-commercial scope |
| KLiCKe | none published | train locally, publish nothing derived from it (this covers the `motor-phase2` checkpoint) |
| How We Type | CC BY-NC 4.0 | finger-label ground truth; cite Feit, Weir & Oulasvirta, CHI 2016 |
| IteraTeR | Apache-2.0 | revision structure for synthetic draft chains (opt-in via `build_dataset.py --iterater`) |

Nothing under `data/`, `checkpoints/`, or `models/` is committed.
