# GPU run chronicle — Lambda 1×H100, 2026-08-11 to 2026-08-13

> **Verdict correction (2026-08-14, after this chronicle closes).** Every
> `pass_model` number in this record is inflated by the writer-identity leak
> found the next day: the held-out pool drew 200 sessions from 14 writers,
> and pair-grouped CV folds put the same writer in train and test. Under the
> corrected writer-grouped protocol (`3ae1a69`, `--max-per-writer`),
> **Tier-1 is met**: `motor-full` 0.5129 and `motor-phase2` 0.5100, all five
> gates (`docs/results-qwen35-4b-gpu.md`). The lever ledger's nulls (data,
> epochs, temperature, RAFT) were measured against that artifact and say
> nothing about the levers; RAFT round 1 remains a null result. The
> narrative below is unchanged: it is the record of how the compute was
> actually spent.

The complete record of the rented GPU box, from provisioning (2026-08-11
~05:00 UTC) to the time of writing (2026-08-13 22:10 UTC, RAFT generation in
flight). All times UTC — the box's clock and every log marker. Sources: the
git history on `feat/data-pipeline-motor-model`, `logs/`, the committed
`eval_report*.json` files, `docs/results-*.md`, the session transcripts on
the box, and filesystem timestamps. Incidents are recorded as they happened;
where a contemporaneous account disagrees with the on-disk evidence, both are
given and the timestamps win.

**The box:** Lambda H100 PCIe 80 GB, Ubuntu 22.04, 26 cores, 221 GB RAM.
**What arrived on it:** the repo at `3371a52` (the clone reflog's 05:01:14
entry; the Mac kept committing past the handoff, through `72cc3f9` at
05:57 UTC, but those commits only reached this line in the Aug 12 merge) —
the Phase-1 motor-model
pipeline built locally over Aug 9–10, a GPU handoff runbook defaulting to
Qwen2.5-7B-Instruct, and the local 0.8B shakedown reports
(`eval_report_08b.json`: 0 of 120 generations valid, 64% malformed — the
grammar-failure baseline every later number is measured against). Commits
timestamped `-0700` throughout this history are the user's local Mac
workstream; commits timestamped `+0000` from `685c59c` (06:05:18) onward are
this GPU session.

---

## 1. Setup day — 2026-08-11

### ~05:00–06:01 — provisioning, and the first dataset build dies

The box came up around 05:00 (repo cloned 05:01, venv built 05:02, test
cache 05:06). Setup, the Aalto corpus fetch, and the first full-corpus
dataset build all started in the first minutes; the Qwen2.5-7B download (the
runbook default, later superseded) completed at 05:58.

The sequential dataset build was the day's first problem. When the GPU
session picked it up at 05:50 it was crawling: polars `group_by` iteration
on tiny per-file frames wasted rayon threads, burning 3.5 cores for 13–14
files/s — ~3.3 hours for the 165,008-file corpus, not the ~25 minutes the
runbook had estimated. The investigation added its own trap: `timeout N sudo
strace` leaked a root-owned strace that `timeout` could not signal; it
stayed attached, throttled the build ~8× further, and blocked py-spy with
EPERM until it was killed by hand at 05:57:26.

At 06:01:35 the build process was found dead — killed by a Ctrl-C in the
tmux pane, at roughly 25% of the corpus, with every parsed row held in
memory and therefore lost. The session's compute ledger booked it as **~70
minutes of CPU lost**; the filesystem bounds it tighter — the corpus
download only finished at 05:04:09, so the build ran at most ~57 minutes.
Because it also exposed the 13–14 files/s rate and
the all-in-memory failure mode, it directly motivated everything in the next
section.

### 06:03–06:41 — split fix, parallel builder, byte-identical validation

Asked how to restart (06:03), the user chose subset-now-plus-full-behind.
Before relaunching, a latent split bug was fixed: `split_by_writer` shuffled
the full writer list and took a prefix, so a subset build and a full build
disagreed about nearly every held-out writer. `685c59c` (06:05:18) made the
split a per-writer hash — stable under subsetting, converging to 10.067%
held-out on the corpus's 165,009 writers — and later verification found **0
disagreements across 17,905 shared writers** between the subset and full
datasets.

At 06:05:32 two sequential builds were launched: a 20k-file subset (done
06:30:22, 24m50s: **236,489 train examples from 16,081 writers; 26,863 test
from 1,824**; 228 sessions dropped) and a full build behind it. While they
ran, the builder was parallelized at the user's prompting (`1d12aae`,
06:38:57): file-level multiprocessing with spawn workers (fork would inherit
polars' live rayon threads) and `POLARS_MAX_THREADS=1`. A validation build
of the same 20k files ran 06:33:11–06:33:35 — **~24 seconds against 24m50s
sequential, ~60×** — and `cmp` confirmed `train.jsonl`, `test.jsonl`, and
`split.json` **byte-identical** to the sequential output. (The commit
message's "~4 minutes" for this build is contradicted by the log timestamps
and by the measured ~1,000 files/s; 24 seconds is the number the evidence
supports.) The still-running sequential full build was killed at 06:39:16
and the parallel full build replaced it: 06:39:16–06:41:29, **2m13s** for
all 165,008 files — **1,948,826 train examples from 132,559 writers; 218,287
test from 14,850**; 1,937 sessions dropped.

### 06:18–06:55 — Qwen3.5-4B (user direction) and the CUDA kernel saga

The runbook said Qwen2.5-7B-Instruct. The run owner overrode it ("its
supposed to be 3.5 that we use"), setting the family to Qwen3.5 and the size
band 2B–4B with the pick delegated; **Qwen3.5-4B** was chosen — 5× the 0.8B
whose local shakedown had shown the data-scaling signal. The model
downloaded 06:18:39–06:18:55; `cc08df8` (06:41:34) made it the default after
verifying on the box that the multimodal wrapper resolves to
`Qwen3_5ForCausalLM` and that `resize_token_embeddings` grows the tied table
248,320 → 260,887 with 12,810 event-token embeddings seeded.

Qwen3.5's hybrid linear attention brought a dependency trap: transformers
disarms its entire fast path unless BOTH `flash-linear-attention` AND
`causal-conv1d` import. Measured without them: training ~5k tok/s (fine),
decode ~15 tok/s (unusable for eval). `flash-linear-attention` installed
cleanly (06:18:57). `causal-conv1d` did not: no wheel exists for torch
2.13.0+cu130, the cu13torch2.10 wheel dies at import (`undefined symbol:
...materialize_cow_storage`), and the system nvcc was CUDA 12.8 against the
venv's cu130 torch. The fix (06:40:40–06:54:38): install `cuda-toolkit-13-0`
from NVIDIA's apt repo and build from source with
`CUDA_HOME=/usr/local/cuda-13.0 CAUSAL_CONV1D_FORCE_BUILD=TRUE` — an 8m36s
compile, then numerical validation against a pure-torch reference (fn err
0.03125, update err 0.03056) before the build was trusted.

### 06:41–09:30 — smoke tests and the epoch-1 subset run

Two training smoke tests ran while the kernel built (06:41:42–06:46:11 and,
at batch 32, 06:48:30–06:50:28; checkpoints `smoke`, `smoke-b32`). Held-out
writer prompts were exported to `data/heldout_writers` (06:43) — the fixed
eval pool every later Tier-1 eval draws from. The real run launched at
06:55:52: LoRA fine-tune on the 236k-example subset, batch 32, peak 24.1 GiB.
Epoch 1 finished at 09:29:45 — **7,391 steps, 2h33m20s, train_loss 2.748**.
Support tooling landed while it ran: a whole-run dashboard (`3374367`, which
paid for itself immediately by surfacing a transient 37s step caused by CPU
contention from review agents), a fix for transformers silently dropping
per-step loss lines when stdout is redirected (`e8840f1`), and the
adversarial-review fixes `a646f70`/`6bac1c0`.

### 09:30–10:16 — the first eval: 1% validity, and the EOS-parity bug

The first Tier-1 eval (09:30:06–10:00:50) was a catastrophe: **2 valid
generations in 200 attempts — 1% validity**, 197 rejected as wrong-text, on
a model whose training loss said it had learned the task. The failure was
temperature-independent — the same pattern at every temperature tried — which
pointed at the mask, not sampling.

The bug (`d51f757`, 10:16:11): the constrained decoder had **EOS parity
inverted**. It allowed EOS after `<DT:>` — the dangling-gap ending
`deserialize()` rejects — and forbade it after an event token, the only
legal stop (the final keystroke has no gap). The trained model put p≥0.9998
on EOS at the true stop point; the mask −inf'ed it, sampling fell into the
residual `<DT:>` mass, and generation typed on past the target until the
token budget died. The inversion had survived since the module was written
because its only tests were network-marked and always skipped, and the
generation test scrubbed trailing `<DT:>` off before parsing — hiding the
exact evidence. The fix was one line; new offline tests drive the mask with
a fake tokenizer and pin both parities. No training run was invalidated —
the model had been fine all along; it was being measured wrong.

### 10:26–10:58 — the honest first eval

Rerun with the fix (10:26:04–10:56:55), `eval_report_qwen35_4b_20k.json`:
**validity 99.5%** (1 wrong-text in 201 attempts), discriminator vs
heuristic baseline 1.00, control 0.40 (band edge), **model 0.895** vs the
≤0.55 gate, serial-dependence 0.4975 — dead chance, because the featurizer's
only order-sensitive feature was a single noisy lag-1 autocorrelation. 3 of
5 gates passed. Marginals: iki/hold/burst KL 0.07–0.09, **pause KL 5.98** —
the pause tail the standout gap.

### 11:29–17:02 — epochs 2–3 (user direction) and the serial-teeth upgrade

Direction from the run owner: two more epochs on the subset before spending
on the full corpus. `--resume` was added (`81a9437`) and training resumed
11:29:47, reaching 22,173 total steps (3 epochs) at 16:29:28 — 4h59m, loss
plateauing ~2.64 by mid-epoch-2. Meanwhile the eval discriminator gained
**nine serial-dependence features** (`ec8a89f`): lag-1/2/3 autocorrelation,
von Neumann ratio, fast/slow Markov excess, local-to-global spread,
half-session drift, hold/gap coupling, word-boundary slowdown. Validated on
200 real held-out sessions: real-vs-shuffled 0.4975 → 0.8075 — the
serial-teeth gate could now pass at all.

The epoch-3 eval (16:30:28–17:01:49, `eval_report_e3.json`): **model 0.8225**
(improved despite the stronger discriminator), serial teeth 0.7975, validity
99.5%, but **pause KL worsened 5.98 → 7.29** — two more epochs sharpened
fluent typing rather than teaching the pause tail. Longer training was
declared a spent lever.

### 17:39–18:11 — the codec comb: 0.8225 → 0.5775, Tier-1 near-miss

Feature-importance analysis over the dumped generations
(`data/generation_dump_e3.jsonl`) found the discriminator was reading **the
codec, not the model** (`c0aa902`): decoded generations take ~38–40 distinct
hold values (the serializer's 128-bin log grid) while raw corpus sessions
take ~223, and the GBM put 0.438 of its importance on a single hold quantile
— a quantization comb no token-emitting model can beat. `run_eval` now
projects every session — real, generated, baseline — through
`codec_roundtrip()` before scoring, reporting the raw-real number alongside.

The codec-fair rerun (17:39:27–18:10:25, `eval_report_e3_codecfair.json`)
is Phase-1's headline:

| Gate | Value | Needs | |
|---|---|---|---|
| generation_validity | 99.5% | ≥90% | pass |
| discriminator_has_teeth | 1.00 | ≥0.90 | pass |
| serial_dependence_teeth | 0.8075 | ≥0.75 | pass |
| control_near_chance | 0.465 | 0.40–0.60 | pass |
| model | **0.5775** | ≤0.55 | **fail by 0.0275** |

Pause KL fell 7.29 → 0.92 under fair scoring — most of the "pause gap" had
been the comb. The residual signal was diffuse (top GBM importance 0.092,
hold/gap coupling). **Tier-1: 4 of 5 gates, near-miss.** Results committed
(`b55e93e`, `a4452b2`, 18:11).

### 18:21–23:19 — the overnight duplicate-launch incident (~5h lost)

At 18:21 the full-corpus overnight chain (train one epoch on 1.95M examples,
then auto-eval) was launched and verified running. The tmux scrollback later
showed something the session record cannot explain: **two ~22 GB training
processes running simultaneously** (PIDs 104328 and 106933) — a duplicate
launch fighting over the GPU at roughly half speed each. One ended, later
the other, and at 23:19 a single clean run was started using the exact
original launch line, its file descriptors pointing at the original log
paths. Shell history was never flushed, so the surviving evidence cannot
establish where the second process came from or who killed what; only two
actors had shell access. **Cause undetermined.** Cost: the restart discarded
~5 hours of training progress from the 18:21 launch, because nothing
checkpoints mid-epoch. The chained script was committed as
`scripts/overnight_full.sh` (`4f950c4`, 23:19:34) and the clean run's first
log marker is 23:19:34. The branch was pushed to GitHub at `4f950c4` that
night, 16 commits ahead of the morning's clone point.

---

## 2. Full-corpus day — Aug 11 23:19 → Aug 12 22:56

### 23:19 → 19:58 — the clean epoch

The relaunched chain trained one epoch over all 1,948,826 examples:
**60,901 steps, 20h22m45s** of training (20h39m wall including load and
label-building; the 23:28 ETA estimate of "~13:30" proved ~6.5 hours
optimistic), **final train_loss 2.601**, token accuracy 0.1891, 2.138×10⁸
tokens — `checkpoints/motor-full`.

While it trained, Phase 2 was prepared and the local workstream arrived.
At 00:18 the Mac side finished uploading **KLiCKe: 4,992 session CSVs +
4,992 gold texts**, verified box-side (file counts exactly matching the Mac;
adapter spot-check 24/30 sessions replaying exactly — expected yield with
the paste/rollover gates), and recorded the licensing gate: **train freely,
publish nothing KLiCKe-derived** until terms are resolved
(`KLICKE_UPLOAD_COMPLETE.md`). The merge `c7a54d4` (00:13 UTC) reconciled
the GPU day with the Mac's tiny-model workstream — the `-0700` commits: the
tiny motor PoC, keyboard-geometry reconstruction/interp work, the live
playground, the harness-ceiling control. Notably, **the two sessions had
independently found and fixed the same two eval bugs** (EOS parity and codec
fairness); the merge kept the GPU side's stricter variants. Combined suite:
206 tests green. The box then added Phase-2 trainer support (`2341d2c`,
01:23): `--init-adapter` (start a new run from a finished checkpoint's LoRA
+ trained embeddings) and `--mode both` (mixed `<MODE:T>`/`<MODE:C>`
curriculum).

### 19:58–21:33 — codec-fair eval, temperature sweep, lever exhaustion

The chained eval (19:58:30–20:29:28, `eval_report_full.json`): validity a
perfect **200/200**, and marginals improved across the board — pause KL
0.92 → **0.46**, iki 0.021 → **0.014**, burst 0.100 → **0.067**, hold
0.051 → 0.047 — **and the gate did not move: model 0.60** (vs 0.5775 on the
subset checkpoint, flat within CV noise on the same held-out pool).
8.25× the data bought distributional realism, not separability.
`timing_only` 0.58 tracked the full number: still caught on timing, not
length.

The temperature sweep closed the last cheap hypothesis (t=0.9:
20:31–21:02; t=0.8: 21:02–21:32): **1.0 → 0.60, 0.9 → 0.66, 0.8 → 0.685**
— monotone. Cooling pulls generation variance below human variance and the
discriminator reads exactly that; the distribution-faithful default was
already the model's best setting.

The Phase-1 lever ledger, as recorded in `docs/results-qwen35-4b-gpu.md`
(`c2cd2be`, 22:56):

| Lever | Result |
|---|---|
| More data (8.25×) | marginals up, gate flat |
| Longer training (3 epochs) | gate flat |
| Lower temperature | actively harmful, monotone |
| Stronger eval (serial features, codec-fair) | made the gate honest |

Tier-1 stands at 4/5 with `pass_model` ~0.58–0.60 against ≤0.55; the
remaining designed route is Phase-3 adversarial polish
(discriminator-guided preference optimization).

### 21:34–21:38 — Phase-2 dataset, and training launched

The parallel builder rebuilt with `--klicke` in ~3 minutes
(21:34:05–21:37:18): **27,480 KLiCKe composition + 32,878 Aalto
transcription examples (2,500-file anti-forgetting fraction), split into
54,328 train examples from 5,792 writers** and 6,030 test from 641, one
writer split across both corpora. Phase-2 training launched at 21:37:44: 3 epochs of the mixed
curriculum, continuing from `checkpoints/motor-full` via `--init-adapter`.

---

## 3. Phase 2 — Aug 12 21:37 → Aug 13, and the Tier-2 arc

### 21:37 → 10:14 — Phase-2 training

**5,094 steps, 12h33m03s, loss 2.94 → 2.569, token accuracy 0.2157** — the
highest of any run — landing at 10:14:07 as `checkpoints/motor-phase2`.

### 10:14–10:32 — sanity probes: behavior right, guarantee missing

With no Tier-2 eval yet in existence, sanity probes ran first
(`scripts/probe_phase2.py`, results `16ec7c5` and
`docs/results-phase2-composition.md`):

- **Transcription regression: 20/20 valid** through the constrained decoder
  — the anti-forgetting mix preserved the motor skill outright.
- **Composition** (5 held-out KLiCKe prompts, deliberately UNCONSTRAINED):
  all 5 streams parsed with zero grammar failures and no mask — against the
  0.8B shakedown's 64% grammar-failure rate. Event mixes tracked per-prompt
  conditioning (a 5.7%-BKSP prompt drew 4.9%; a 20.9% prompt drew 24.2%);
  CURSOR and SELDEL both appeared; think-pauses present but thin (~50–70% of
  the real >1s fraction — the tail-thinness Phase 1 measured); realistic
  typo-revision texture ("To begin gin," then a revision region; uncorrected
  "cando").
- **Two defects, both decoding-side:** no convergence pressure — only 2/5
  sessions started exactly on target, and free generation **composes its own
  on-topic essay** when probability drifts (KLiCKe taught "write an essay
  like this"; nothing at sampling time forces the given text) — and
  unvalidated cursor ops (2/5 emitted "cursor 10 outside buffer of length
  1", which replay correctly rejects).

### 10:39–11:36 — the convergence decoder: 1/5 → cooldown fix → 5/5 → 50/50

Both defects had one answer: the design-spec §6 buffer-tracking constrained
decoder (`e12186b`, 10:39): a `ConvergenceProcessor` that maintains the live
TextBuffer as tokens commit; in event position the needed on-path key is
always sampleable, any key while the excursion budget lasts (typos stay
real), BKSP always, and at the budget BKSP only; in gap position every
`<DT:>` (timing stays fully model-sampled) and EOS **iff buffer == target**.

The first live probe on the phase-2 checkpoint **failed 4/5** (converged
1/5): with excursions perpetually open, the model — which wants its own
essay — typed wrong, was forced back, typed wrong again: ~50% BKSP, token
budget exhausted. The fix (`069f173`, 10:50) made convergence structural: a
resolution cooldown arming on ANY on-path backspace (not just budget-forced
ones), with guard state derived strictly from the committed token stream, so
every resolution cycle nets progress — proven by a test whose adversarial
sampler always prefers wrong keys and still terminates on the exact target.
The re-probe: **5/5** (done 10:55:26). Scaled validation launched at
10:55:58 across 50 held-out prompts: **50/50 exact** (done 11:35:51) — with
fully realistic texture, e.g. session 49: 348 events, 18.4% BKSP,
converging to the exact target.

### 11:13–11:14 — Tier-2 modules, built by parallel agents

Three independent modules were built in parallel by subagents and
integration-reviewed adversarially (`8155af1`, suite 251 passed):
`eval/signatures.py` (spec §8.4 composition signatures — pause-at-boundary
IKIs classified from the live buffer state, P-burst lengths, revision-op
statistics), `eval/knobs.py` (spec §8.3 requested-vs-realized knob fidelity,
Pearson + MAE per knob), and `converge.py` stage 2 — `<CUR:>`/`<SELDEL:>`
re-admitted under the guarantee via a digit-level state machine in which
every digit is masked so the growing number stays completable to a position
valid for the LIVE buffer ("cursor 10 in a 1-char buffer is unrepresentable,
not unlikely"). The Tier-2 eval runner assembling them landed as `871563c`
(11:14:33), and a chained smoke-then-full eval was armed at 11:14:34 behind
the still-running n=50 validation. At 11:27 the overnight RAFT chain
(section below) was armed behind that, and the box was declared self-driving
for the user's ~10 hours away.

### 11:36–11:43 — the watcher-deadlock incident

The Tier-2 chain's wait condition was `while pgrep -f converge_probe; do
sleep 30; done` — wait for the n=50 probe to exit. But the background
watchers armed to *follow* that same probe carried `converge_probe` in their
own command lines, so the wait condition could never clear: a circular wait
between the chain and its watchers. The probe finished at 11:35:51; the
chain should have fired within 30 seconds and instead sat blocked — forever,
had it not been caught. It was diagnosed at 11:42 and manually unblocked at
11:43:26; the smoke eval fired at 11:43:36.

Honest accounting: the end-of-day report to the user described this as "~40
minutes" of idle caught at an hourly check. The surviving timestamps bound
the actual GPU-idle window at **~8 minutes** (11:35:51 → 11:43:36), caught
in a routine post-commit check, not an hourly one; the same report's
timeline shifted the morning's events ~40–60 minutes late throughout. The
mechanism description ("its own command line matched the chain's wait
condition") was accurate.

### 11:43–16:50 — the first Tier-2 composition eval

Smoke, n=10 (11:43:36–12:08:32): 10/11 attempts converged (rate 0.909),
harness sane. Full, n=100 (12:08:33–16:50:33, 4h42m,
`eval_report_composition.json`): **convergence 76.3%** (100 scored in 131
attempts), `tier2_provisional_met: false` — and the numbers localized the
gap precisely:

- **Harness healthy:** serial teeth 0.99, control 0.49.
- **Timing nearly solved:** hold KL 0.008, pause KL 0.026, all three
  pause-position classes (clause/word/within-word) at 0.13–0.17. The model
  pauses like a writer.
- **The discriminator wins at 0.965 on REVISION behavior:** `bksp_frac` KL
  4.1, `cursor_count` KL 7.1, `seldel_count` KL 4.9 — under the
  stage-1-dominant mask the model barely produces cursor/seldel ops, and its
  backspace fraction is distorted by the mask's resolution discipline. The
  realism gap moved from timing to behavior.
- **Composition knob fidelity weak** (WPM r=0.43, MAE 8.7) against a
  near-perfect transcription control run: **WPM r=0.994, MAE 2.5** — the
  knobs work; composition's horizon breaks them.

The convergence misses and the knob weakness shared one root, diagnosed and
fixed the same evening (`22164ca`, 21:29): composition was trained on
≤512-event windows with a `<WRITTEN>` tail, and a 400-char essay needs 800+
events in one shot — past its trained horizon the model degrades, fights the
mask, and burns the token budget. `generate_windowed()` re-prompts per
window exactly as training did, pre-seeds the ConvergenceProcessor buffer,
and opens continuation windows in gap slot so the model emits the
window-boundary `<DT:>` itself. A windowed re-validation was chained behind
the night's work (below). Results and the KLiCKe upload record were
committed as `d50dba3` (21:58).

### 16:51 → in flight — RAFT: the Phase-3 polish begins

With every sampling- and scale-side lever exhausted, the night chain moved
to the designed remaining route: **RAFT preference-data generation**
(`gen_raft_data.py`, `6ff2fdd`) — K candidates generated per
TRAINING-writer target, scored by a discriminator trained only on training
writers (the holdout appears nowhere), keeping the most human-scoring
candidate as SFT data; conditioning labels are inverted from each stored
prompt and asserted byte-identical, so every winner was generated under
exactly the conditioning it is paired with.

- **Smoke** (16:51:17–16:55:32): 10 targets, K=2 — discriminator paired CV
  0.70 on train writers, 10 winner examples written. Passed.
- **Full** (16:55:34, IN FLIGHT at time of writing): **800 targets, K=4**
  from `checkpoints/motor-full`. At 21:57:28 it stood at 500/800 targets
  (2,000 candidates); at ~36s/target the generation finishes roughly 01:00
  Aug 14. The chain then automatically fine-tunes
  (`--init-adapter motor-full`, 2 epochs on the winners →
  `checkpoints/motor-raft`), re-runs the Tier-1 eval on the same 200-session
  held-out pool → `eval_report_raft.json`, and only then hands the GPU to
  the windowed Tier-2 re-eval → `eval_report_composition_windowed.json`.

---

## 4. External datasets (Aug 13, background agent)

While the GPU ran evals, a background agent expanded the corpus base
(`d448627`, 11:41):

- **How We Type** (CHI 2016, CC BY-NC 4.0, Zenodo 4034268) — needed no
  request; downloaded to `data/raw/howwetype/`. Adapter built with the
  standard integrity gates: 98.1% byte-exact replay, the 2
  rollover-corrupted logs dropped, plus `iter_sessions_with_fingers()` —
  every keypress carries its ground-truth finger, turning the reconstruction
  workstream's same-finger inference into a supervised check. The bytes
  contradict the official readme in four documented ways (epoch seconds,
  key-downs only, lowercase + Shift rows, CSV quoting required). 17 fixture
  tests; suite 268 green.
- **IteraTeR** human subset (Apache-2.0, ~5.5 MB) — downloaded to
  `data/raw/iterater/`: 4,018 labelled edits, 559 document revisions, 145
  real multi-depth draft chains. `docs/iterater-notes.md` maps
  `edit_actions` char offsets onto CURSOR/SELDEL/KEY events as grounding for
  the spec §4.1.3 synthetic revision trajectories; no adapter yet, per plan.
- **Request drafts** (`docs/dataset-requests.md`): ready-to-send emails for
  the two request-only corpora — Clarkson II (via CITeR; 103 users, 12.9M
  uninstructed longitudinal keystrokes) and Buffalo CUBS (148 subjects, 73
  crossing keyboards between sessions).

---

## 5. Compute ledger

| When (UTC) | Item | Duration | Outcome |
|---|---|---|---|
| Aug 11 ~05:04–06:01 | Sequential dataset build #1 (CPU) | ≤~57 min (booked at the time as ~70) | **Lost** — tmux Ctrl-C at ~25%; motivated the 60× parallel builder |
| Aug 11 06:05–06:41 | Rebuilds: subset (24m50s seq), par validation (24s), full (2m13s) | ~28 min | Banked — both datasets + byte-identical proof |
| Aug 11 06:40–06:54 | causal-conv1d source build (CPU) | 14 min | Banked — fast path armed |
| Aug 11 06:41–06:50 | Training smoke tests | ~7 min | Banked |
| Aug 11 06:55–09:29 | Subset epoch 1 | 2h33m | Banked — `motor-e1` |
| Aug 11 09:30–10:00 | Eval #1 (broken: 1% validity) | 31 min | Half-banked — its failure pattern exposed the EOS bug |
| Aug 11 10:26–10:56 | Eval #2 (fixed) | 31 min | Banked — first honest Tier-1 report |
| Aug 11 11:29–16:29 | Subset epochs 2–3 | 4h59m | Banked — `motor` |
| Aug 11 16:30–18:10 | Epoch-3 eval + codec-fair eval | ~1h2m | Banked — the 0.5775 near-miss |
| Aug 11 18:21–~23:15 | Overnight launch #1 | ~5h | **Lost** — duplicate-launch tangle, cause undetermined; nothing saves mid-epoch |
| Aug 11 23:19–Aug 12 19:58 | Full-corpus epoch (clean) | 20h39m wall / 20h23m train | Banked — `motor-full` |
| Aug 12 19:58–21:32 | Full eval + temperature sweep (t=0.9, 0.8) | ~1h34m | Banked — gate flat; temperature proven a non-lever |
| Aug 12 21:37–Aug 13 10:14 | Phase-2 mixed-curriculum training | 12h36m wall / 12h33m train | Banked — `motor-phase2` |
| Aug 13 10:14–10:32 | Sanity probes | ~18 min | Banked |
| Aug 13 10:40–11:35 | Convergence probes (1/5, 5/5, 50/50) | ~55 min | Banked — decoder validated at scale |
| Aug 13 11:36–11:43 | Watcher deadlock | **~8 min idle** (reported at the time as ~40) | Lost |
| Aug 13 11:43–16:50 | Tier-2 smoke + full eval | 5h7m | Banked — first Tier-2 report |
| Aug 13 16:51–16:55 | RAFT smoke | 4 min | Banked |
| Aug 13 16:55– | RAFT full generation | 5h15m so far | **In flight** |

Totals across the rental: **~5.1 GPU-hours and ~57 CPU-minutes lost to two
incidents plus one ~8-minute stall**; everything else banked. No training
run was ever invalidated by a bug — all three Phase-1 bugs (EOS parity,
codec comb, order-blind discriminator) were in the eval and tooling, all
pre-dated the rental, and all are now pinned by offline tests.

---

## 6. State at time of writing — 2026-08-13 22:10 UTC

**In flight, self-driving:**
- `gen_raft_data.py --targets 800 --k 4` (PID 182620): 500/800 targets as of
  21:57; H100 at ~22 GB, 36% util. Chained behind it (`night_chain.sh`):
  RAFT SFT → `checkpoints/motor-raft` → Tier-1 re-eval →
  `eval_report_raft.json`; expected to land roughly 01:30–02:30 Aug 14.
- `windowed_chain.sh` armed behind the night chain: windowed Tier-2 eval
  n=100 → `eval_report_composition_windowed.json` — the direct test of
  whether `22164ca` recovers the 23.7% convergence shortfall and the
  composition knobs.

**Checkpoints on disk:** `motor-e1` (7.6G, subset epoch 1), `motor` (28G,
subset epoch 3 — Tier-1 4/5 at 0.5775), `motor-full` (18G, full-corpus epoch
— 0.60, the motor foundation and the RAFT base), `motor-phase2` (28G, the
composition model), `smoke`/`smoke-b32` (18G each, disposable), `motor-raft`
pending.

**Committed record:** branch `feat/data-pipeline-motor-model`, pushed
through `d50dba3`. Ten committed eval reports in the repo root, with the
logs, chart the arc: 0% validity (0.8B local) → 1% (EOS bug, log only) →
99.5%/0.895 → 0.8225 → 0.5775 codec-fair → 0.60 full → 0.66/0.685 (sweep) →
Tier-2 0.909-smoke/0.763-full.
Untracked leftovers: `eval_report.json` and `eval_report_e3.json` (byte
duplicates of the committed `_qwen35_4b_20k*` reports) and `logs/`.

**Open gates:** Tier-1 `pass_model` 0.58–0.60 vs ≤0.55 — RAFT is the live
attempt. Tier-2 provisional unmet — revision behavior (cursor/seldel under
the mask) is the localized gap; windowed generation is the queued fix for
convergence and knobs; stage-2 cursor ops exist but need the model to use
them. **Licensing:** nothing KLiCKe-derived may be published until terms are
resolved — this includes `motor-phase2` and every composition artifact.

**What lands next**, in order: `eval_report_raft.json` (did
discriminator-guided preference data move the last Tier-1 gate?), then
`eval_report_composition_windowed.json` (does horizon-matched generation fix
convergence and knob fidelity?), then the request emails for Clarkson II and
Buffalo CUBS await the user's send.
