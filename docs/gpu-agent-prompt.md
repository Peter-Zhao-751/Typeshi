# Prompt for Claude Code on the GPU box

Copy everything below the line into a fresh Claude Code session running in
`~/Typeshi` on the rented GPU instance.

---

You are continuing work on **Typeshi**, a project that trains a language model
to generate realistic human typing *processes* — not just final text, but the
keystroke-by-keystroke stream with real timing, typos, and corrections.

The data pipeline and evaluation harness are finished and tested. **Your job is
the Phase-1 motor fine-tune: train the model, evaluate it, and iterate until
the Tier-1 milestone passes or you can explain precisely why it cannot.**

## Read these first

1. `docs/gpu-handoff.md` — the runbook you are executing. Start here.
2. `docs/superpowers/plans/2026-08-09-data-pipeline-and-motor-model.md` — the
   full implementation plan. Tasks 1–12 are all complete and committed;
   Task 10 step 4 (the training run) and Task 12 step 4 (scoring a real
   checkpoint) are what remain.
3. `docs/data-schemas.md` — the real corpus schemas. The published papers and
   the plan's guesses were both wrong about KLiCKe; this file is ground truth.
4. `docs/training-on-apple-silicon.md` — why this moved to a GPU, and
   measurements you may find useful for comparison.

## What the system does

A keystroke session is a list of canonical `Event`s (`KEY`, `BACKSPACE`,
`CURSOR`, `SELDEL`). Events serialize to a token grammar the model learns:

```
<DT:12><KEY:h><HOLD:7><DT:9><KEY:i><HOLD:6><DT:44><BKSP><HOLD:5>
```

`DT` is press-to-press delta, `HOLD` is press-to-release, both quantized into
128 log-spaced bins between 1 ms and 120 s. Log spacing gives millisecond
resolution on fast keystrokes and second-scale resolution on thinking pauses.
Rollover — a key pressed before the previous is released — shows up implicitly
as `HOLD > DT`, which is why `DT` is never negative.

Training examples are `{"prompt", "completion"}` pairs. The prompt carries the
target text plus condition knobs (`MODE`, `WPM`, `ERR_COR`, `ERR_UNC`, `REV`);
the completion is the serialized event stream.

## Your task, in order

1. Run `bash scripts/setup_gpu.sh`. It provisions everything and ends with the
   test suite green and `data/processed/split.json` written.
2. Smoke-test training (`--epochs 0.002`). Verify the log reports
   `bf16': True`, `seeded 354 event-token embeddings`, and a falling loss.
3. Train. **Start with a subset** — rebuild with `--limit-aalto 20000` for
   ~230k examples. A full epoch is ~400M tokens and Phase 1 only needs motor
   timing, which likely saturates far earlier. Scale up only if eval demands.
4. Run `scripts/run_eval.py` and read the report.
5. Iterate toward the gates. If the model fails, the levers in plan order are:
   more data, lower sampling temperature, longer training, then a
   sequence-model discriminator to find what summary statistics miss.

## The Tier-1 gates

From `eval_report.json`:

- `pass_discriminator_has_teeth`: accuracy **≥ 0.90** separating real sessions
  from the naive heuristic baseline. This proves the discriminator works at
  all. It scored 1.00 on real Aalto data locally, so if it drops, suspect the
  eval before celebrating.
- `pass_model`: accuracy **≤ 0.55** separating real sessions from our
  generations. This is the actual goal — the discriminator *should* fail here.
- `discriminator_accuracy_real_vs_real_control`: should sit near chance. If it
  is high, the writer population is separable and `pass_model` is inflated.

## Things that will bite you if you do not know them

- **Python 3.14 has no stable torch wheels.** The setup script pins 3.12. Do
  not "helpfully" upgrade it.
- **The eval refuses to run without `data/processed/split.json`.** That is
  deliberate. Holdout is by *writer*, never by session, and scoring realism on
  writers the model trained on would pass Tier-1 for the wrong reason. Do not
  reach for `--allow-unsplit` to make an error go away.
- **`modules_to_save` is load-bearing.** The 356 event tokens are new; LoRA
  alone never touches the embedding table, so without this they keep their
  initial vectors forever and `lm_head` can never learn to emit them. Only use
  `--freeze-embeddings` if you genuinely run out of memory, and report it.
- **Timing features must be pooled per session**, via `compare_sessions`, not
  by concatenating events. Every session is rebased to zero, so a flat
  concatenation invents a large negative inter-key interval at each boundary
  and makes the Fréchet distance NaN.
- **Sessions that fail exact replay are dropped, never patched.** ~4% of
  KLiCKe has cursor positions corrupted by key rollover. Exact replay is the
  correctness gate; do not relax it to raise yield.
- **`generate()` can emit text outside the grammar**, which raises
  `ValueError`. `run_eval.py` counts these as
  `generations_rejected_as_malformed`. A high count is a real finding about
  model quality — report it, do not silence it.

## How to work

- Run `python -m pytest -q` before and after changes. It should stay green
  (101 passed, 3–4 skipped). Network-dependent tests need
  `TYPESHI_NETWORK_TESTS=1`.
- Commit as you go, following the existing message style in `git log`.
- Long runs: use `tee` to a file under `logs/` and check in periodically
  rather than blocking on them.
- **Report honestly.** If Tier-1 fails, say so with the numbers. A truthful
  failing report is the useful outcome here; a passing number obtained by
  loosening a gate is worse than useless, because the whole point of this
  milestone is knowing whether the output is actually indistinguishable from
  real typing.

## Constraints from the plan

- English only; desktop/physical keyboards only.
- Integer milliseconds everywhere internally, never floats for timestamps.
- All randomness takes an explicit seed; default 0.
- Every module has a matching test that needs neither network nor real corpora.

## Licensing

Aalto is non-commercial research use with attribution (Dhakal et al., CHI
2018). Do not set up any commercial deployment of a checkpoint trained on it.
