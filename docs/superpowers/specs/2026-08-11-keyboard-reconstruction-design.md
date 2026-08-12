# Keyboard Reconstruction from the Tiny Motor Model — Design

**Date:** 2026-08-11
**Status:** Approved direction (locked in brainstorming); pending user review of
this written spec.

## 1. Goal

Recover the physical structure of a QWERTY keyboard from `motor-tiny` — a model
that has never seen a key coordinate, only event streams of the form
`<a:54><DT:46><s:54>`.

The claim under test: **physical layout is the latent variable generating the
timing, and a model trained only on the timing has to represent some of it.**
Press-to-press gaps are driven by hand alternation, same-finger conflicts, row
jumps, and travel distance; a model that predicts `<DT:k>` well must encode
those regularities somewhere.

The `<DT:k>` softmax makes this unusually direct. No trained probe is needed to
read the model's beliefs about motor cost — the 128-way distribution *is* the
belief, and it can be read at any position by teacher-forcing.

### What a positive result establishes

- The dataset's **timing signal carries recoverable physical structure**, and
  the model learned it. This is evidence about the data that does not route
  through the discriminator, and so survives the unresolved harness-ceiling
  question in `harness_control.json` (see `2026-08-10-tiny-motor-poc-design.md`
  §5a) — the realism gates currently cannot distinguish "generator is bad" from
  "featurization is broken", and this probe sidesteps that entirely.
- A 19M from-scratch model at this data scale is learning motor structure, not
  just copying text.

### What it does not establish

- Nothing about generation realism, the five gates, or Phase 1's GPU run.
- Nothing about composition mode — `motor-tiny` is trained on `<MODE:T>` only.

### Non-goals

- Attention/circuit analysis, steering vectors, the playground visualization.
  Those were considered in brainstorming and deferred; they are separate specs.
- Any change to training, eval, serialization, or the grammar.

## 2. Expected shape of the result — revised against measurement

The original version of this section predicted "a motor distance, not Euclidean
geometry": strong hand separation, decent finger-column structure, noisier rows.
Before Task 5 was built, the corpus was measured directly (all 1.97M sessions,
82.8M digraphs). The prediction was right in direction and **much too optimistic
in degree**, so it is replaced here by what was actually found.

### 2.1 Two-dimensional geometry is a null

| reconstruction | ρ vs true coordinates |
|---|---|
| synthetic matrix (pipeline ceiling, §7) | 0.89 |
| corpus, all 27 keys | **0.03** |
| corpus, left hand only (15 keys) | 0.31 |
| corpus, right hand only (11 keys) | 0.21 |

The pipeline is not at fault — Task 2 proves it recovers a known keyboard at
0.89. The signal is not present in the form MDS requires, and §2.3 explains why.

### 2.2 What the corpus does support

- **The biomechanical ordering**, with no modelling at all:
  `repeat 155 ms < alternate 197 < same_hand 217 < same_finger 226`.
- **Same-finger detection**: AUC rises from 0.59 at support ≥ 20 to **0.90** at
  support ≥ 20,000. The weak figure was sampling noise, not absent structure.
- **A conditional distance law** (§2.3), the strongest and most interpretable
  result available.

### 2.3 Distance is conditional on the hand relationship

Regressing the double-centred residual on key-to-key distance *within* each
motor class:

| class | slope (log-ms per key-width) | correlation |
|---|---|---|
| alternate | −0.001 | −0.01 |
| same_hand | +0.063 | +0.25 |
| same_finger | +0.214 | +0.43 |

Travel distance is free when the hands alternate — the hands move in parallel —
costs something when one hand must travel, and costs most when one finger must.
This is textbook motor biomechanics recovered from nothing but token timing, and
it is **why no single metric space can embed the matrix**: "distance" denotes
three different quantities depending on the hand relationship. A global MDS is
the wrong instrument for the board as a whole, which is exactly what the 0.03
records.

### 2.4 The central question for the model probe

The corpus's structural terms explain **R² = 0.05** of the digraph residual; the
other 95% is writer and context heterogeneity. The model, queried with a fixed
carrier and fixed knobs, returns a *conditional expectation* — denoised by
construction. The prediction, now sharp and testable:

> the model should deliver the high-support answer for **every** cell, including
> the rare bigrams the corpus can never measure at any sample size.

Confirming or refuting that is the point of Tasks 5 and 6. It is also the only
justification for probing a model rather than reading the corpus directly.

## 3. Components

New package `src/typeshi/interp/`, plus `scripts/keyboard_probe.py` as the
entry point. Nothing here imports from or modifies the training or eval paths.

CPU by default, for the reason `playground.py` documents: a training run owns
the MPS device and a probe that steals it would slow the real work.

### 3.1 `layout.py` — ground truth

Pure data, no model. Key centers for the 26 letters plus space on a standard
ANSI staggered layout, in key-width units (1u = one key width), y increasing
upward:

| Row | Keys | x of first key | y |
|---|---|---|---|
| Top | `Q`–`P` (10) | 2.00 | +1 |
| Home | `A`–`L` (9) | 2.25 | 0 |
| Bottom | `Z`–`M` (7) | 2.75 | −1 |
| Space | `SPC` | 6.875 | −2 |

Successive keys in a row are 1.0u apart. Offsets are the real ANSI stagger
(Tab 1.5u, Caps 1.75u, LShift 2.25u; space is 6.25u starting at x=3.75).

Also the standard touch-typing assignment table:

| Hand | Finger | Keys |
|---|---|---|
| Left | pinky | `q a z` |
| Left | ring | `w s x` |
| Left | middle | `e d c` |
| Left | index | `r f v t g b` |
| Right | index | `y h n u j m` |
| Right | middle | `i k` |
| Right | ring | `o l` |
| Right | pinky | `p` |
| — | thumb | `SPC` |

Derived bigram classes, used throughout: `repeat` (same key twice), `same_finger`
(same finger, different key), `same_hand` (same hand, different finger),
`alternate` (different hands). `repeat` is a separate class deliberately —
double-taps are fast and would otherwise contaminate the same-finger class,
which is the slowest.

### 3.2 `digraph.py` — the model probe

Produces a 27×27 matrix of predicted press-to-press latencies in ms (26 letters
+ space, ordered pairs; the diagonal is the `repeat` class and is kept).

**Carrier.** One fixed target sentence with the bigram at a fixed character
offset, so every cell differs *only* in the two characters:

```
<TARGET>the ratios{a}{b} in the sample<PROCESS>
```

Everything before `{a}` is constant, which is what makes the 729 probes a
controlled contrast — the thing a model can give you and an observational
corpus cannot. Knobs are fixed: `<MODE:T><WPM:10><ECOR:0><EUNC:0><REV:0>`
(bin 10 ≈ 50–55 wpm, near the corpus mode).

The insertion point deliberately follows a **letter** rather than a space. With
a `… of {a}{b} …` frame, the 27 cells where `{a}` is space would produce a
double space and read as a different motor event than the rest of the matrix;
following `ratios` keeps the space row on the same footing as every other row.

The cost: pairs like `qz` never occur in English and the model is extrapolating
there. That is what §3.3 exists to bound. Two matrices, and their disagreement
is diagnostic.

**Prefix identity.** The teacher-forced completion prefix — the serialization of
typing `the ratio of ` — is decoded **once** from a reference prompt and then
reused byte-identically for all 729 probes. Decoding it per pair would let the
prompt's differing bigram leak into the prefix's hold bins. Pinned by a test,
not a comment.

**Hold bin for the `{a}` press.** Per-character modal hold bin measured from the
corpus (realistic and character-specific), with a sensitivity check at two fixed
global bins — the corpus-wide modal hold bin and the corpus-wide median hold
bin, applied uniformly to every character. Reconstruction metrics must not move
materially across the three; if they do, that is reported rather than the best
run being picked.

**Readout.** Forward once per pair; at the position of the `<a:h>` token take
the logits, restrict to the 128 `<DT:k>` IDs (the grammar guarantees a DT token
comes next), renormalize, and summarize as the **geometric mean** over
`from_bin` centers — `exp(Σ p_k · log center_k)`. Arithmetic mean is the wrong
average over geomspaced bins. Batched; ~730 forwards is seconds on CPU.

### 3.3 `empirical.py` — the data baseline

The same 27×27 matrix measured directly from `train.jsonl`. Parse completions
with `serialize._TOKEN_RE`, walk consecutive KEY events, and record
`(prev_char, char, dt_bin)` only where the two presses are adjacent in the
stream — pairs separated by a `<BKSP:h>` are skipped. Summarized as the
**geometric mean** over `from_bin` centers — `exp(mean(log t))` — which is the
same functional as the model readout in §3.2 and so directly comparable; the
median is computed alongside as a robustness check. A support count per cell is
recorded so thin cells can be masked rather than silently averaged.

Sampled at 200k lines (of 1.98M) for turnaround; the sample is seeded and the
count per cell is recorded, so coverage is auditable rather than assumed.

This is the ceiling: if the corpus does not encode QWERTY, the model cannot.

### 3.4 `reconstruct.py` — matrix → layout → score

1. **Symmetrize** in log-ms: `D[a,b] = (L[a,b] + L[b,a]) / 2`.
2. **Remove main effects** by iterated two-way mean centering, NaN-aware.
   Load-bearing: some keys are simply slow, and without stripping row and
   column effects the first component is "fast keys vs slow keys" and geometry
   never surfaces. What remains is the interaction term, which is where layout
   lives.

   This step originally specified Tukey median polish, for robustness against
   the same-finger outliers. Measured against the synthetic matrix of §7 that
   choice was wrong: median polish recovers ρ=0.61 where mean centering
   recovers ρ=0.90, against a ceiling of 0.97. At 27 columns the median is too
   high-variance to preserve a distance signal this small, and the outliers it
   was guarding against are removed explicitly one step later.

   Robustness therefore lives in the **masking**, not the estimator. Every
   stage is NaN-aware, and masking up to 60 of the 351 unordered cells still
   recovers ρ=0.85. The honest limitation: an *unmasked* gross outlier does
   degrade the fit (12 cells at 10× drop it to ρ=0.57), so `mask_thin_cells`
   and the log-space averaging upstream are what keep the estimator's
   assumptions true, and both are pinned by tests.
3. **Same-finger handling**, two variants, always reported side by side:
   - **Blind** (uses no ground truth): same-finger pairs are *detected* as the
     high-positive-residual outliers of the interaction matrix — median + 2·MAD,
     the standard robust upper-tail rule — and that detected set is regressed
     out before reconstruction. The detection is itself scored against truth
     (§6 level 2), so this stays non-circular.

     A two-component Gaussian mixture was specified first and measured much
     worse: ρ=0.58, flagging ~140 pairs against 41 true ones. Only 12% of the
     351 unordered pairs are same-finger, and a minority bump that small does
     not split a broad continuous residual into two clean Gaussians — the
     mixture cuts the bulk roughly in half. The MAD rule lands at 44
     detections (precision 0.77, recall 0.82) without being told the count.
   - **Finger-aware** (uses ground truth): regress out the true same-finger
     indicator. Diagnostic only — it answers "is the geometry there once the
     biomechanical confound is accounted for", and must be labelled as
     ground-truth-assisted wherever it appears.
4. **Embed**: shift residuals to non-negative, metric MDS (SMACOF) to 2D.
5. **Align**: Procrustes onto the true coordinates, reflection allowed.

## 4. Controls

| Control | Rules out |
|---|---|
| Random-init model, same config, same probe | The probe or the analysis manufacturing structure from nothing |
| Frequency regression — strip log unigram and bigram frequency (computed from the corpus `<TARGET>` texts, no external dependency) and redo | "It learned English", i.e. layout inferred from letter statistics rather than motor cost |
| Empirical baseline (§3.3) | Attributing to the model what is merely in the data |
| Permutation test on the Procrustes fit | Reading a good-looking alignment as significant when chance would do it |

## 5. Sweeps

Once single-checkpoint scoring works, both are re-runs of the same harness:

- **WPM conditioning** — two well-populated bins, `<WPM:5>` (≈25 wpm) and
  `<WPM:16>` (≈80 wpm). Hand-alternation structure should dominate more at
  speed, so the reconstruction should sharpen. A world model that gets crisper
  under load is a stronger claim than a static map.
- **Checkpoint series** — every snapshot in `checkpoints/interp-snapshots/`
  (preserved out from under `save_total_limit=2` by a watcher started
  2026-08-11). Metrics versus step, plus a small-multiples figure: the keyboard
  condensing out of noise.

## 6. Scoring

Four levels, ordered by how well the corpus measurement (§2) says each holds.
Every level is reported for the model, the corpus, and the random-init control
side by side — the comparison, not the absolute number, is the result.

1. **Biomechanical table** — mean latency by bigram class (`alternate` <
   `same_hand` < `same_finger`, with `repeat` fast and separate). Robust, needs
   no MDS. **Holds in the corpus** at 155/197/217/226 ms.
2. **Same-finger detection AUC** — how well the interaction residual ranks true
   same-finger pairs. Needs no MDS, no alignment, no threshold. **Holds**, and
   is support-dependent: 0.59 → 0.90 as cells become well-measured. The primary
   test of the §2.4 denoising prediction, since the model should reach the
   high-support figure on *all* cells rather than only frequent ones.
3. **Conditional distance law** — the distance slope fitted separately within
   `alternate`, `same_hand` and `same_finger`, plus each class's correlation.
   **Strongly present** in the corpus (−0.001 / +0.063 / +0.214). A model that
   reproduces this ordering has learned the biomechanics, not just the marginals.
4. **Full geometry** — Spearman correlation of pairwise distances, mean position
   error in key-widths, nearest-neighbour recall, each with the permutation
   test, for the whole board and per hand. **Expected null** (§2.1: corpus 0.03
   against a 0.89 ceiling). Reported anyway, beside the synthetic ceiling, so
   the null is legible as a property of the data rather than of the analysis.

Levels 1–3 carry the evidence. Level 4 is reported for completeness and is
expected to fail; presenting it as anything else would misrepresent the finding.

### 6a. What the pipeline can deliver, measured

Established on the synthetic matrix before any model was loaded, and the
numbers every model result gets read against:

| quantity | measured | meaning |
|---|---|---|
| ceiling, true distances fed straight in | ρ = 0.97 | the most MDS + Procrustes can return |
| finger-aware reconstruction | ρ = 0.89–0.90, error 0.77u | geometry is recoverable once the biomechanical confound is accounted for — **uses ground truth** |
| blind reconstruction | ρ = 0.80–0.85 | the price of using no ground truth at all |
| same-finger AUC | 0.98 | **the strong claim**: the residual ranks same-finger pairs to the top |

The AUC is the result that carries the evidence — it needs no MDS, no
alignment, and no threshold. The blind 2D map is the weakest of the three and
should be presented as such.

Output per run: a metrics JSON alongside the existing `eval_*.json` convention,
and a figure.

## 7. Testing

- **Pipeline recovery test (the one that matters).** Generate a *synthetic*
  latency matrix from the true coordinates plus a known same-finger penalty and
  known main effects, run it through `reconstruct.py`, and assert it recovers
  the layout to a tight tolerance under both variants. Without this a null
  result is uninterpretable — it cannot be attributed to the model rather than
  to the analysis.
- **Layout table** pinned: key count, row membership, finger assignment, and
  the derived class of a handful of hand-checked bigrams (`ed` same-finger,
  `th` alternate, `er` same-hand, `ss` repeat).
- **Prefix identity** pinned: the teacher-forced prefix token IDs are identical
  across a sample of pairs (§3.2).
- **Readout** pinned: the geometric-mean summary matches a hand-computed value
  on a synthetic distribution, and the DT ID set has exactly 128 members.
- **Empirical parser** pinned against a fixture completion with a known bigram
  and a known backspace-separated pair that must be skipped.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Recovered map is motor distance, not geometry | Predicted in §2; scored at three levels (§6); blind and finger-aware variants reported together (§3.3) |
| Synthetic carrier is out of distribution for rare bigrams | Empirical baseline bounds it (§3.3); disagreement reported rather than hidden |
| Structure is letter frequency, not layout | Frequency regression control (§4) |
| Analysis code manufactures a keyboard | Synthetic recovery test (§7) and random-init control (§4) |
| Thin corpus support for rare cells | Per-cell counts recorded and masked, not silently averaged |
| Probe steals MPS from the live training run | CPU default, per `playground.py`'s precedent |
| Same-finger detection is circular | Blind variant detects from the residual distribution and is scored against truth (§3.3, §6.2) |

## 9. Build order

1. `layout.py` + table tests. **Milestone: hand-checked bigram classes pass.**
2. `reconstruct.py` + the synthetic recovery test. **Milestone: a keyboard is
   recovered from a synthetic matrix.** This lands before any model is loaded —
   it is what makes a later null result mean something.
3. `empirical.py` + parser test. **Milestone: measured corpus digraph matrix,
   with the biomechanical table (§6.1) computed from data alone.**
4. `digraph.py` + prefix-identity and readout tests. **Milestone: model matrix
   for one checkpoint.**
5. `scripts/keyboard_probe.py` wiring, controls (§4), metrics JSON + figure.
   **Milestone: all three scoring levels reported for `step-15000`, against the
   random-init and empirical baselines.**
6. Sweeps (§5). **Milestone: WPM contrast and the checkpoint series.**
7. Results doc alongside `results-08b-shakedown.md`.
