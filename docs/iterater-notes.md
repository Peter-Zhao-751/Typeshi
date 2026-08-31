# IteraTeR — human edit-intention dataset (field guide; adapter: `src/typeshi/adapters/iterater.py`)

Downloaded 2026-08-13 to `data/raw/iterater/` from HuggingFace
(`wanyu/IteraTeR_human_sent`, `wanyu/IteraTeR_human_doc`; ~5.5 MB total).
Paper: Du, Raheja, Kumar, Kim, Lopez & Kang, *Understanding Iterative
Revision from Human-Written Text*, ACL 2022 (arXiv 2203.03802).
**License: Apache-2.0** per the dataset cards — no non-commercial restriction,
unlike the keystroke corpora.

The **human-annotated subset only** was taken, per plan. The machine-annotated
releases exist on the same account and were *not* downloaded:
`IteraTeR_full_sent` (~166 MB), `IteraTeR_full_doc` (~125 MB),
`IteraTeR_v2` (~432 MB, 24k docs / 170k edits).

Files are **JSON Lines despite the `.json` extension** (one object per line).

## Row counts and fields

| Repo | train | dev | test | total | unit |
|---|---|---|---|---|---|
| `human_sent` | 3,254 | 400 | 364 | 4,018 | one labelled sentence-level edit |
| `human_doc` | 481 | 27 | 51 | 559 | one full document revision (draft depth *n* → *n+1*) |

The two views are the same annotations: `human_doc`'s 4,018 `edit_actions`
flatten exactly into `human_sent`'s 4,018 rows.

**`human_sent` fields:** `before_sent`, `before_sent_with_intent` (before-text
prefixed with an `<intent>` tag — their model's input format), `after_sent`,
`labels` (single intention label), `doc_id`, `revision_depth`.

**`human_doc` fields:** `doc_id`, `revision_depth` (1–4), `before_revision`,
`after_revision` (mean 1,241 / median 1,076 / max 9,028 chars), `domain`
(`news` 189, `wiki` 170, `arxiv` 162, missing 38), `sents_char_pos` (sentence
boundaries), and `edit_actions` — a list of
`{type: R|A|D, before, after, start_char_pos, end_char_pos, major_intent,
raw_intents}`. Char offsets index into `before_revision`, so each action is
directly executable against the draft text. Edit-op types over all 4,018
actions: **R**eplace 2,769, **A**dd 663, **D**elete 586.

## Edit-intention label distribution (all splits, n=4,018)

| Label | n | share | typical shape |
|---|---|---|---|
| `clarity` | 1,601 | 39.8% | local rewording, deletion of redundancy |
| `fluency` | 942 | 23.4% | grammar/typo/spelling fixes — small span replacements |
| `meaning-changed` | 896 | 22.3% | content added or altered — the big insertions |
| `coherence` | 393 | 9.8% | connectives, reordering, cohesion edits |
| `style` | 128 | 3.2% | tone/voice changes |
| `others` | 58 | 1.4% | formatting etc. |

`revision_depth` distribution (sent-level): depth 1 = 2,917, depth 2 = 904,
depth 3 = 172, depth 4 = 25. Across the 427 unique documents, 145 have real
multi-step chains (125 reach depth 2, 18 depth 3, 2 depth 4).

## How this grounds §4.1.3 synthetic revision trajectories

The design spec (§4.1 item 3) wants backward-constructed draft chains
`D0 → … → Dn = T`: an LLM writes plausible earlier drafts, diffs become
edit-op streams, and the motor model assigns timings. IteraTeR is the
empirical anchor for every step of that pipeline:

1. **Real chains as seed trajectories.** The 145 multi-depth documents are
   *actual* `D0 → D1 → … → Dn` chains. Each `edit_actions` list is already a
   mid-level edit-op stream with character offsets: an `R` at
   `[start_char_pos, end_char_pos)` maps directly onto our canonical events as
   CURSOR(start) → SELDEL(start, end) → KEY(after-chars); `A` is CURSOR + KEY
   run; `D` is CURSOR + SELDEL. Feed those through the phase-1 motor model to
   assign keystroke timings and we have complete synthetic composition
   sessions grounded in real human revisions — no LLM draft-writer needed for
   this slice.
2. **A rejection filter for LLM-generated drafts.** The backward-chained
   drafts the spec proposes have no ground truth. IteraTeR gives the target
   statistics a plausible chain must match: op-type mix (69% replace / 16%
   add / 15% delete), span-length distributions, edits-per-pass (~7 per
   document revision), and how edit density falls with revision depth
   (2,917 → 904 → 172 → 25). Synthetic chains whose diffs diverge from these
   marginals get rejected before entering fine-tuning.
3. **Intention labels as generation conditioning.** The label distribution is
   a natural mixture prior for the draft-writer prompt ("introduce a fluency
   error", "remove a clarification") and can surface as conditioning tokens
   (e.g. `<INTENT:clarity>`) alongside the existing knob values, letting the
   phase-2 model learn that fluency edits are small local replacements while
   meaning-changed edits are multi-clause insertions.
4. **What it does not give.** No within-pass ordering (edits in one revision
   are simultaneous diffs — the order to execute them must come from KLiCKe's
   observed cursor-movement patterns), no timings at all (by design; the
   motor model owns those), and formal written domains (arxiv/news/wiki) that
   skew more polished than typical live composition. Treat it as structure
   ground truth, not behavior ground truth.

The adapter now exists (2026-08-25): `adapters/iterater.py`, with
`adapters/timing.py` drawing hold/gap pairs from real KLiCKe pools, parses
`edit_actions` into canonical CURSOR/SELDEL/KEY event streams and exports a
concatenable shard via `build_dataset.py --iterater`. See
`docs/open-work.md` (2026-08-25 section) and `docs/revision-fix-runbook.md`
for the export gates and yield.
