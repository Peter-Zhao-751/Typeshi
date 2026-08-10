# Token Format v2

Canonical spec for the serialization in `src/typeshi/serialize.py`. Supersedes
the v1 grammar in the original plan (`docs/superpowers/plans/…`) — v1 was never
trained on, so no compatibility path exists or is needed.

## An example, end to end

Prompt (input to the model; loss is **not** computed on it):

```
<MODE:T><WPM:11><ECOR:0><EUNC:5><REV:0><TARGET>I was trying to remember what Bobby said.<PROCESS>
```

Completion (what the model learns to emit):

```
<I:56><DT:54><SPC:53><DT:56><w:53><DT:59><a:49><DT:49><d:51>...
```

Every token above is a **single vocabulary entry** — the strings are only how
we spell them for humans and the parser; the model sees integer IDs.

## Event grammar

| Token | Count | Meaning |
|---|---|---|
| `<c:h>` | 97 × 128 | Key `c` pressed, held for time-bin `h` |
| `<BKSP:h>` | 128 | Backspace, held for bin `h` |
| `<DT:k>` | 128 | Press-to-press gap of bin `k` to the **next** event |
| `<CUR:p>` | regex | Caret moved to offset `p` (unbounded int → plain text) |
| `<SELDEL:a-b>` | regex | Range `[a,b)` deleted (unbounded ints → plain text) |

- **Stream shape:** `event (<DT:k> event)*` — one keystroke is two tokens.
- **No leading `<DT:0>`.** It was byte-identical in 100% of 400k examples.
  A *continuation window* may open with `<DT:k>` carrying the real gap across
  the window boundary (`serialize(events, prev_press_time=...)`).
- **Char identities (97):** 92 printable ASCII directly, plus five named
  escapes `SPC NL TAB LT GT` (space, newline, tab, `<`, `>`).
- All time bins live on **one shared scale**: 128 log-spaced bins over
  1 ms – 120 s (`src/typeshi/timebins.py`).

### Rollover

```
<X:y><DT:z>   with y > z   →   X was still held when the next key went down
```

A plain integer comparison, valid **only because holds and gaps share the
same bin scale** — that is why HOLD did not get its own compact scale, even
though 99.6% of holds occupy just 24 bins. 26% of real keystrokes overlap
this way, so the property matters.

## Prompt grammar

```
<MODE:m><WPM:w><ECOR:e><EUNC:u><REV:r><TARGET>{text}[<WRITTEN>{text}<CUR:p>]<PROCESS>
```

| Token | Count | Binning |
|---|---|---|
| `<MODE:m>` | 2 | `T` transcription, `C` composition |
| `<WPM:w>` | 40 | 5-wpm buckets; corpus max is 156, never clamps |
| `<ECOR:e>` `<EUNC:u>` `<REV:r>` | 31 each | whole %-points, clamped at 30 |
| `<TARGET>` `<WRITTEN>` `<PROCESS>` | 3 | structure markers |

- The **target text stays natural language** — the one place the base model's
  pretraining earns its keep (and the spec's justification for using an LLM
  at all: plausible earlier drafts in composition mode).
- `<WRITTEN>…<CUR:p>` appears only on continuation windows: the buffer content
  and caret position where this window resumes.
- There is **no instruction boilerplate** ("Simulate the writing process…"):
  it cost 10 constant tokens in every one of 2M examples.

## Vocabulary

12,810 registered tokens (+ 2 regex prefixes `<CUR:`, `<SELDEL:`):

| Group | Tokens |
|---|---|
| `<c:h>` char × hold | 12,416 |
| `<BKSP:h>` | 128 |
| `<DT:k>` | 128 |
| knobs + markers | 138 |

At Qwen2.5-7B's hidden size that is ~92M new embedding parameters (both
matrices) — 8% of the base vocabulary. Combinations that never occur in
typing simply never appear in training; no fallback scheme is needed.

## Measured, v1 → v2 (extended tokenizer, real transcription examples)

| | v1 | v2 |
|---|---|---|
| prompt tokens | 52.0 | **16.7** |
| completion tokens | 143.9 | **91.4** |
| total | 195.9 | **108.1 (−44.8%)** |
| loss computed on | prompt + completion | **completion only** |

The loss change matters as much as the length change: v1 collapsed both into
one text field, so ~27% of the training signal was spent predicting the prompt.

## Design decisions and their evidence

| Decision | Evidence |
|---|---|
| Merge char+hold into one token | −32% sequence length; `MI(char;hold)=0.073` bits, so it is a pure length-for-vocab trade at trivial vocab cost |
| Keep DT separate | Full merge needs 95×128×128 ≈ 1.5M entries (5.6B params); DT is also the highest-entropy component (4.79 bits) and deserves its own softmax |
| One shared bin scale | Keeps `y > z` rollover comparison valid; per-bin resolution near 112 ms is ~1.10× either way, so nothing is lost |
| All 128 hold bins per char, no fallback | The fallback saved 36M params (0.5% of the model) for a permanent special case; scrapped |
| Drop leading `<DT:0>` | Constant in 400,000/400,000 examples |
| Carry window-boundary DT | v1 zeroed one real gap per 512 events — in composition, potentially a minutes-long pause |
| Knobs as single tokens | v1's text header: 28 tokenizer pieces for five numbers |
| Keep 128 DT bins | Composition uses 112/128 including the clamped top bin |
| `<CUR:>`/`<SELDEL:>` stay regex text | 0.86% of composition events, 0% of transcription |
