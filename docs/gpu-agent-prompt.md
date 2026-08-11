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
2. `docs/token-format.md` — the token grammar (format v2) and the
   measurements behind every design decision. **The original plan documents
   show the older v1 grammar; this file supersedes them.**
3. `docs/superpowers/plans/2026-08-09-data-pipeline-and-motor-model.md` — the
   full implementation plan. Tasks 1–12 are all complete and committed;
   Task 10 step 4 (the training run) and Task 12 step 4 (scoring a real
   checkpoint) are what remain.
4. `docs/data-schemas.md` — the real corpus schemas. The published papers and
   the plan's guesses were both wrong about KLiCKe; this file is ground truth.
5. `docs/results-08b-shakedown.md` — the local 0.8B baseline you are
   beating: the data-scaling curve (0% -> 14.2% valid from 3.8k -> 17.7k
   examples), and the MPS sampler bug that corrupted earlier numbers.
6. `docs/training-on-apple-silicon.md` — why this moved to a GPU.

## What the system does

A keystroke session is a list of canonical `Event`s (`KEY`, `BACKSPACE`,
`CURSOR`, `SELDEL`). Events serialize to a token grammar the model learns
(format v2 — full spec in `docs/token-format.md`):

```
<h:51><DT:49><e:51><DT:50><SPC:51><DT:44><BKSP:48>
```

One keystroke is two tokens: `<c:h>` (the key plus its hold-time bin) and
`<DT:k>` (press-to-press gap to the next event). Holds and gaps share one
128-bin log-spaced scale over 1 ms – 120 s, which is what makes rollover a
plain comparison: `<X:y><DT:z>` with `y > z` means X was still held when the
next key went down — true of 26% of real keystrokes.

Training examples are `{"prompt", "completion"}` pairs:

```
<MODE:T><WPM:11><ECOR:0><EUNC:5><REV:0><TARGET>the sentence here<PROCESS>
```

Knobs and markers are single registered tokens; the target text stays natural
language. TRL masks the prompt from the loss — only the event stream trains.

## Your task, in order

1. Run `bash scripts/setup_gpu.sh`. It provisions everything and ends with the
   test suite green and `data/processed/split.json` written.
2. Smoke-test training (`--epochs 0.002`). Verify the log reports
   `bf16': True`, `seeded 12810 event-token embeddings`, and a falling loss.
3. Train. **Start with a subset** — rebuild with `--limit-aalto 20000`
   (~235k examples). The local scaling curve (0% -> 14.2% valid going
   3.8k -> 17.7k examples) says data volume is the binding lever, so expect
   to scale up quickly if validity tracks that curve; a full epoch is ~215M
   tokens.
4. Run `scripts/run_eval.py` and read the report.
5. Iterate toward the gates. If the model fails, the levers in plan order are:
   more data, lower sampling temperature, longer training, then a
   sequence-model discriminator to find what summary statistics miss.

## The Tier-1 gates

From `eval_report.json` — ALL five must hold (`tier1_met`):

- `pass_discriminator_has_teeth`: **≥ 0.90** vs the naive heuristic baseline.
  Scored 1.00 on real Aalto data locally; if it drops, suspect the eval.
- `pass_serial_dependence_teeth`: **≥ 0.75** vs timing-shuffled real sessions.
- `pass_model`: paired grouped-CV accuracy in **[0.40, 0.55]**. Below 0.40 is
  leakage, never realism — unpaired CV scored 0.085 on EXACT COPIES, which is
  why `train_discriminator(paired=True)` exists. Never score paired data with
  unpaired folds.
- `pass_generation_validity`: **≥ 90%** of generation attempts parse and
  actually type the target. A model judged only on its rare successes is not
  being judged.
- `pass_control_near_chance`: real-vs-real in [0.40, 0.60].

Also compare `discriminator_accuracy_vs_model_timing_only` against the full
number: a big gap means length is doing the separating, not timing.

## Things that will bite you if you do not know them

- **Python 3.14 has no stable torch wheels.** The setup script pins 3.12. Do
  not "helpfully" upgrade it.
- **The eval refuses to run without `data/processed/split.json`.** That is
  deliberate. Holdout is by *writer*, never by session, and scoring realism on
  writers the model trained on would pass Tier-1 for the wrong reason. Do not
  reach for `--allow-unsplit` to make an error go away.
- **`modules_to_save` is load-bearing.** The 12,810 event tokens are new; LoRA
  alone never touches the embedding table, so without this they keep their
  initial vectors forever and `lm_head` can never learn to emit them. Only use
  `--freeze-embeddings` if you genuinely run out of memory, and report it.
- **Timing features must be pooled per session**, via `compare_sessions`, not
  by concatenating events. Every session is rebased to zero, so a flat
  concatenation invents a large negative inter-key interval at each boundary
  and makes the Fréchet distance NaN.
- **Sessions that fail integrity checks are dropped, never patched.** KLiCKe:
  rollover-corrupted cursor positions (~4%) and paste/drag sessions (13.8% —
  a paste becomes zero-IKI "keystrokes" that poison motor timing). Aalto:
  sessions replaying under 0.90 similarity to USER_INPUT, and sessions with
  characters outside the 97-identity English vocabulary. Do not relax any of
  these to raise yield.
- **Sampling is Gumbel-argmax, not torch.multinomial** (typeshi/constrain).
  multinomial on MPS emitted tokens with verified -inf logits; Gumbel is
  distributionally identical, immune by construction, and the default on
  every device. Do not "simplify" it back to do_sample=True.
- **Constrained decoding is on by default in the eval** and eliminates
  grammar failures by construction; `--unconstrained` is the diagnostic path.
- **`generate()` can emit text outside the grammar** (unconstrained mode),
  which raises
  `ValueError`. `run_eval.py` counts these as
  `generations_rejected_as_malformed`. A high count is a real finding about
  model quality — report it, do not silence it.

## How to work

- Run `python -m pytest -q` before and after changes. It should stay green
  (138 passed, a handful skipped). Network-dependent tests need
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
