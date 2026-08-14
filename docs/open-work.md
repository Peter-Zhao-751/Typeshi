# Open work — what to fix next, and what it needs

State as of 2026-08-14 05:30 UTC, written for continuing on a laptop after
the rented GPU is released. The narrative record of how we got here is
`docs/gpu-run-chronicle.md`; this file is only what remains.

**Legend:** 💻 = no GPU needed (do this locally) · 🖥 = needs a GPU box.

The split is between *implementing* and *measuring*. Every fix below can be
written and unit-tested on a laptop; Fixes 1, 3 and 4 then need one GPU
eval run to confirm the effect, and Fix 5 needs two or three. Only Fix 2
needs the GPU for the work itself. The efficient pattern is therefore:
do all the local work first, then rent a box once and run every pending
measurement in a single session.

## Where the project stands

| Model | What it is | Verdict |
|---|---|---|
| `motor-full` | 1 epoch, full 1.95M-example Aalto corpus | **Best measured transcriber.** Tier-1 4/5 gates; discriminator 0.600 vs the ≤0.55 gate; 100% validity |
| `motor-phase2` | `motor-full` + KLiCKe composition (mixed curriculum) | Composition-capable, transcription retained 20/20. Tier-2 scored (below) |
| `motor-raft` | `motor-full` + RAFT round 1 | **Dead end.** 0.6075, pause KL regressed 0.46 → 1.22 |
| `motor`, `motor-e1` | 20k-subset checkpoints | Superseded; keep only for provenance |

Tier-1 fails on one gate only: a discriminator distinguishes generated from
real typing 60% of the time (gate: ≤55%). Everything else passes, including
the two teeth checks and the control.

Tier-2 (first-ever run, `eval_report_composition.json`, single-shot
decoding): timing is nearly solved — hold KL 0.008, pause 0.026, all three
pause-position classes 0.13–0.17 — while the discriminator wins at 0.965 on
**revision behavior** (see Fix 1).

## Fix 1 💻 — the decoder suppresses the revision behavior the eval then penalizes

**This is the highest-value fix and needs no GPU.**

The Tier-2 discriminator's signal is not timing; it is revision statistics:
`cursor_count` KL 7.10, `seldel_count` 4.91, `bksp_frac` 4.11, against
timing KLs of 0.008–0.026. The obvious reading is "the model doesn't know
how to revise." That reading is wrong, and the evidence is in the
unconstrained probes (`docs/results-phase2-composition.md`): generating
with the mask OFF, the model produces cursor ops at 0.4–0.8% of events
against real writers' 1.1–1.3%, and emits SELDELs. **It revises; the mask
stops it.**

Root cause, `src/typeshi/converge.py`, the EVENT-position branch: the
`<CUR:`/`<SELDEL:` openers are appended to `allowed` only inside
`if excursions_open:` — the same gate as typo excursions
(`depth < budget and not resolving and cooldown == 0`). While typing
on-path, which is most of any stream, a cursor move is unrepresentable.
There is a second restriction in the same branch: the needed key is offered
only when `cursor == n` (caret at the end), so the mask actively steers the
caret to the end and keeps it there.

Why the gate is over-strict: **a CUR op changes no text, so it cannot break
convergence.** It was gated with excursions because an unbounded caret
walk never terminates — but the correct bound is a cap on *consecutive
cursor ops without text progress*, not a ban on the whole class.

Proposed fix:
1. Allow the `<CUR:` opener on-path, independent of `excursions_open`,
   whenever `depth == 0` (the buffer is a clean prefix of the target).
   SELDEL stays gated — it deletes text and so belongs with excursions.
2. Add a `caret_moves_without_progress` counter, reset on any KEY that
   advances on-path length; when it hits a small cap (2–3), mask cursor
   openers out until progress resumes. This preserves the existing
   termination argument, which currently leans on `max_new_tokens` for
   caret-only samplers (module docstring, "Termination").
3. Keep the deadlock escape hatches (forced cursor-to-end in the two
   resolution corner states) exactly as they are.

Validation without a GPU: extend `tests/test_converge.py` with (a) an
on-path state offering a `<CUR:` opener, (b) the counter capping repeated
caret hops, (c) the existing adversarial walk still converging. Then, when
a GPU is next available 🖥, rerun `scripts/run_eval_composition.py` and
check whether `cursor_count`/`seldel_count` KL collapse and the 0.965
discriminator accuracy drops with them.

Honest caveat: this closes a *measurement* artifact. It may reveal that the
model's revision behavior is merely close rather than right — but the
current number cannot tell us, because the mask is what produced it.

## Fix 1b 💻 — "types the whole line, then deletes it all back to one typo"

Observed while using the model interactively: it makes an early mistake,
keeps typing to the end, then backspaces the entire line to reach the typo,
fixes it, and retypes everything. No human does this. It is a decoder bug,
and it compounds with Fix 1.

Cause, verified directly: `_prefix_edit_depth` measures **edit distance to
the nearest target prefix**, and that does not grow as you type past a
divergence. With target `"hello world this is a longer sentence…"` and the
model having typed `"helo"` and continued correctly:

| chars typed | edit-distance depth | chars past divergence |
|---|---|---|
| 4 | 1 | 1 |
| 20 | 1 | 17 |
| 47 | **1** | **44** |

`excursion_budget` is 4, so resolution never fires — depth stays at 1
forever. The error is only forced at the very end, when EOS is blocked
because `buffer != target`. And because cursor ops are gated behind
`excursions_open` (Fix 1), the only repair route available is
backspace-all-the-way-back.

This is a stage-2 regression. Stage 1 measured depth as
`len(text) - common_prefix_len(text, target)` — the "chars past divergence"
column — which *would* have hit the budget four characters after the typo
and forced a local fix. Stage 2 generalized depth to edit distance so that
legitimate mid-buffer repairs would not count as off-path, which is correct
for that purpose but removed the pressure to correct promptly.

Do **not** simply revert to the suffix rule: during a legitimate cursor
repair the text is briefly far from a prefix (`"helo world"` has suffix
distance 7 while its edit distance is 1), so a suffix-based budget would
forbid exactly the repair it wants to encourage.

Proposed fix — measure *staleness*, not distance:
1. Keep `_prefix_edit_depth` for the on-path test (`depth == 0`) and for
   deciding whether resolution is possible.
2. Add `events_since_on_path`, incremented per emitted event and reset
   whenever `depth == 0`. When it exceeds a small window (8–12 events —
   roughly when a human notices), force repair moves only.
3. "Repair moves" must include cursor ops toward the divergence point, not
   just BKSP; otherwise the fix reproduces the same delete-everything
   behavior. This is why Fix 1 lands first.
4. Termination still holds: the repair set always contains BKSP, which
   strictly shrinks the buffer, and the empty buffer is on-path.

Test it offline by asserting that a sampler which types one wrong character
and then continues is forced into repair within the window, and that the
resulting stream's backspace run is short rather than the length of
everything typed since the mistake.

## Fix 1c 💻 — semantic revisions are forbidden, not absent

Observed in interactive use: the model never writes something plain, thinks
better of it, and replaces it with something better. It only ever makes and
fixes typos.

That is the mask, by construction. `excursion_budget` is **4 characters** —
sized for a motor slip. A semantic revision means typing 10–50 characters
of deliberately non-target text before replacing it, so the mask forces
resolution at character five and the revision can never start. The model is
not the limit: unconstrained it produces 24% backspaces, cursor ops and
SELDELs (`docs/results-phase2-composition.md`).

This is a gap in our implementation against our own design. Spec §6 asks
for "typos **and semantic-revision excursions** (off-target wording)
allowed up to a budget"; only the typo half was built, and the number 4 was
chosen while fighting the oscillation bug, where a wide budget collapsed
convergence to 1-in-5.

### The design: affordability, not a constant

Raising the constant is not the answer, and neither is removing it. An
unbounded excursion deletes the only force pulling generation back to the
target — that is precisely the pre-decoder behavior we measured, where the
model composed its own essay and reached EOS wrong or never (2/5 exact).
A constant budget, meanwhile, is wrong at both ends: 4 forbids revision, 60
would let a wandering sampler burn the whole allowance before anyone
notices.

The rule that dissolves the tension: **allow an excursion of any length,
provided the remaining generation budget can still pay to undo it and
finish the target.** Convergence stays guaranteed because it stays
*reachable* at every step, not because the model is kept on a short leash.

    excursion_allowed  ⟺  cost(resolve current excursion)
                          + cost(type the remaining target)
                          ≤ remaining event budget − margin

Two consequences fall out, and they are what make this practical:

- **SELDEL makes long revisions cheap.** Undoing a 50-character excursion
  costs 50 backspaces, but one cursor move plus one SELDEL is 2 events.
  With Fix 1's ops available, a phrase-level rewrite is affordable early
  in a session and the check above rarely binds. Without them it never is.
  This is why Fix 1 is the prerequisite, not a nicety.
- **The leash tightens by itself.** As the budget depletes the affordable
  excursion shrinks to zero, so the model revises early and polishes late
  — which is what writers actually do, and which gives termination for
  free rather than by fiat.

Pace the *starts* separately from the *length*: open a revision excursion
only at a token boundary and at a rate consistent with the prompt's
`<REV:>` knob, which currently conditions nothing in composition (Fix 4).
Keep the small typo budget alongside it for character-level slips, which
should be able to fire anywhere, not just at boundaries.

One thing this design does **not** give you: a reason for the excursion to
be a *plausible earlier draft* rather than arbitrary off-target text. The
mask can only permit; the content comes from the model. That is exactly the
gap IteraTeR was downloaded to fill (`docs/iterater-notes.md`) — real
human draft→revision chains as training signal, so the detours look like
drafts instead of noise.

Expect this to move the Tier-2 revision statistics that currently dominate
the composition discriminator (`cursor_count` KL 7.10, `seldel_count`
4.91, `bksp_frac` 4.11). Fixes 1, 1b and 1c are one thread: a mask tuned
for transcription typos, applied to composition.

Validation: the oscillation guard is what makes a wide budget safe, so keep
`tests/test_converge.py`'s adversarial-sampler test as the gate, and re-run
the convergence probe (50/50 exact before this change) to confirm the wider
budget does not reintroduce the 1-in-5 collapse.

## Fix 2 🖥 — preference optimization, done properly this time

RAFT round 1 is a measured null: 800 targets × 4 candidates, discriminator
picks the winner, 2 epochs SFT → 0.600 → 0.6075, and pause KL regressed
(`eval_report_raft.json`, commit `55a68cb`).

Why it failed, and what that implies:
- **Best-of-4 cannot remove a shared bias.** All four candidates come from
  one distribution; if every candidate is systematically off in
  hold/gap coupling, selecting among them changes nothing about that.
- **The training signal was faint**: 800 examples over ~50 optimizer steps
  (89 seconds of training).
- Selection also pulled toward the discriminator's *marginal* preferences,
  which is the likely source of the pause-KL regression.

A real attempt should change the mechanism, not the dosage:
1. **Ranked pairs, not winners.** Use `trl`'s `DPOTrainer` with
   (chosen, rejected) pairs drawn from the same target — the gradient then
   sees what makes one *worse*, which is the information best-of-K throws
   away. Prompt format is unchanged; the pair is two completions.
2. **Much larger K** (16–32) so the selected tail is genuinely far from the
   mean, and 5–10k targets rather than 800.
3. **Refresh the discriminator between rounds.** A fixed scorer is a fixed
   target to overfit; retrain it on the new generations each round.
4. **Guard the marginals.** Gate each round on distributional KLs not
   regressing (the RAFT round would have failed such a gate at pause 1.22).

**Cost, measured rather than guessed — and the prerequisite nobody
noticed.** RAFT round 1 generated 3,200 candidates (800 targets × K=4) in
**8h03m** (16:55:34 → 00:58:40): ~400 candidates/hour. At that rate the
5k × K=16 round above is **200 GPU-hours** — roughly $500 and eight days.
Not viable, and an earlier draft of this file put it at "10–20 hours",
which was wrong by an order of magnitude.

The reason is that `gen_raft_data.py` generates **one candidate at a
time**. But the K candidates for a target all share an identical prompt,
which is exactly the case batching handles: same prompt length means the
transcription grammar processor's position parity is shared across rows
(`generated = input_ids.shape[1] - prompt_len`, and `mask[:, allowed] = 0`
is already batch-shaped), and `GumbelSampleProcessor` draws
`torch.rand(scores.shape)` so every row gets independent noise. Both
existing processors are batch-safe for this pattern **today** — only the
driver loop assumes one sequence.

(`ConvergenceProcessor` is *not* batch-safe — it carries a per-sequence
buffer — but RAFT is transcription-mode and doesn't use it. Batching
composition generation is a separate, larger job.)

So the real prerequisite is 💻 **batched candidate generation**: emit K
candidates per `model.generate` call. Expect ~10× throughput at K=16
(not 16× — overhead), turning 80,000 candidates into roughly 20 GPU-hours.
Write and test this locally *before* renting anything; a preference round
run at today's throughput would be an underpowered repeat of the
experiment that already returned null.

Preparation that needs no GPU 💻: batched generation, the pair-construction
script, and the DPO training entry point, with offline tests on synthetic
candidate sets, so the GPU is rented only for the run itself.

**No one can promise this round works.** The honest prior is poor: four
independent levers (8× data, 3× epochs, temperature, best-of-4 selection)
all returned null on this gate, and the residual signal is diffuse — no
discriminator feature above 0.092 importance. What the design above buys
is not a guarantee but a *decisive* result: enough statistical power to
distinguish "preference optimization doesn't move this" from "our
preference round was too weak to tell", which round 1 could not do.
Preregister the stopping rule before spending: one round, and it must
improve paired CV accuracy **and** not regress any distributional KL
(the RAFT round would have failed that second condition at pause 1.22).

## Fix 3 💻/🖥 — the residual convergence failures

Windowed generation took convergence from 76% to ~92% (`22164ca`). The
remaining ~8% is **not** a length problem: the eval caps every target at
400 chars and every KLiCKe test essay exceeds that, so all targets are
exactly the same size — length is constant and cannot separate successes
from failures.

Diagnostics are now in place (`1032d37`): `generate_windowed` reports
on-path progress per window and whether the last window stalled, and
`run_eval_composition` records each failure with the conditioning knobs
that produced it, in the report's `failures` array. **The next composition
eval run answers this**; nothing to guess at now.

Leading hypothesis to test against those records: failures cluster on
high `revision_rate` conditioning, where the model wants to edit more than
the excursion budget permits. If so, the fix is to scale
`excursion_budget` with the requested revision rate rather than holding it
at 4 — plausibly the same change as Fix 1, since both are the mask being
stricter than the behavior it is trying to permit.

## Fix 4 💻 — composition knob fidelity

Composition knob fidelity is weak: WPM r = 0.43, error-rate knobs ≈ 0
(`eval_report_composition.json`). The control says this is not a broken
mechanism: on transcription, the same measurement gives **WPM r = 0.994,
MAE 2.5 wpm**. Re-measure after Fixes 1 and 3; if it stays weak once the
mask stops distorting the event mix, it becomes a training question
(knob-conditioned composition examples are far fewer than transcription's).

## Fix 5 💻 — Tier-2 gate thresholds are deliberately unset

`scripts/run_eval_composition.py` reports signature and knob numbers
without pass/fail, on purpose: a threshold invented before its distribution
is known is a threshold tuned to pass. Once two or three honest Tier-2 runs
exist (post-Fix-1), set thresholds from the real-vs-real control
distribution the way the Tier-1 gates were set, and document the derivation
in `docs/token-format.md`'s style — measurement first, number second.

## Corpora status

- **How We Type** ✅ downloaded, adapter + 17 tests committed (`d448627`).
  Its per-keystroke finger labels are ground truth for the
  keyboard-reconstruction workstream's same-finger rule — that validation
  is available now, locally.
- **IteraTeR** ✅ downloaded (human subset), `docs/iterater-notes.md` maps
  its edit offsets onto CURSOR/SELDEL/KEY events for grounding the spec
  §4.1.3 synthetic revision trajectories. No adapter yet 💻.
- **Clarkson II, Buffalo CUBS** ⏳ request-only; ready-to-send drafts are in
  `docs/dataset-requests.md`. These need a human to send them.
- **Aalto Mobile 37k** — deliberately skipped (v1 non-goal).

## Constraints that must not be relaxed

- **KLiCKe has no license terms.** Nothing KLiCKe-derived — including
  `motor-phase2` — may be published until that is resolved.
- The eval scores real and generated sessions **both** through
  `codec_roundtrip` (`codec_roundtripped_real: true`). Removing that
  re-introduces a 0.22-accuracy quantization artifact no token-emitting
  model can beat.
- Reports written before commit `b55e93e` are not comparable to later ones
  (scoring semantics changed).
- Writer splits are hash-based and stable under subsetting; the eval must
  keep honoring the split bound into each checkpoint.
