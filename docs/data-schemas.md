# Observed Corpus Schemas

Ground truth for the adapters in `src/typeshi/adapters/`. Everything here was
read off the actual downloaded files, not from the papers. Regenerate the raw
listing with `uv run python scripts/fetch_data.py data/raw`.

**Do not code against the published papers' column names — they do not match.**

## License

Neither download ships a license or terms file inside the archive, so the
permissions below could not be verified from the bytes themselves.

| Corpus | Distribution | Status |
|---|---|---|
| Aalto 136M Keystrokes | `userinterfaces.aalto.fi/136Mkeystrokes/` — download gated behind an on-page research-use agreement | Terms accepted at download time on the site; no copy retained in the archive |
| KLiCKe | Google Drive link from `github.com/terryyutian/KLiCKe-Corpus` | No terms file in the archive |

**Open item for the user:** both corpora are published for academic research
use, but neither archive contains a written grant covering *model training*
specifically. Confirm the on-site terms permit training before any checkpoint
is published or served. This does not block local development.

---

## KLiCKe

Source: `data/raw/klicke/KLiCKe corpus/`.

### Layout

| Path | Contents |
|---|---|
| `WritingTask/keystrokelogs/csv/<writer>.csv` | 4,992 composition logs, one per writer |
| `WritingTask/texts/<writer>.txt` | 4,992 final essays — **the replay ground truth** |
| `WritingTask/holistic_scores.csv` | `ID, Prompt, Text, Score`; `Text` duplicates the essay |
| `TypingTests/csv/<writer>_TYPING<n>.csv` | 34,920 transcription logs, same schema |
| `demographic_info.csv` | Per-writer demographics |

**The writer ID is the filename, not a column.** There is no participant column
anywhere in the log files. `iter_sessions` must derive the writer from the path.

### Columns

One row = one key event (or mouse event). Identical across WritingTask and
TypingTests.

| Column | Dtype | Meaning |
|---|---|---|
| `` (unnamed) | Int64 | 1-based row index |
| `DownEventID` / `UpEventID` | Int64 | Event counters |
| `DownTime` | Int64 | Press time, **ms since session start** (not epoch) |
| `UpTime` | Int64 | Release time, ms |
| `ActionTime` | Int64 | `UpTime - DownTime` |
| `DownEvent` / `UpEvent` | String | Key identity: literal char, or a name (`Space`, `Backspace`, `Shift`, `ArrowLeft`, `Leftclick`, …) |
| `CursorPosition` | Int64 | Caret offset **after** the row's edit is applied |
| `PauseTime` | Int64 | Gap since previous event |
| `WordCount` | Int64 | Running word count |
| `TextChange` | String | Text inserted/removed, or `NoChange` |
| `Activity` | String | What the row did — see below |

### `Activity` vocabulary

Counts from a 200-file sample of WritingTask.

| Value | Count | Meaning |
|---|---|---|
| `Input` | 579,316 | Insert `TextChange` |
| `Remove/Cut` | 88,392 | Delete `TextChange` |
| `Nonproduction` | 66,921 | No text effect (modifiers, arrows, clicks) — but `CursorPosition` still moves |
| `Replace` | 427 | Typed over a selection; `TextChange` is `"<old> => <new>"` |
| `Paste` | 35 | Multi-char insert, behaves like `Input` |
| `Move From [a, b] To [c, d]` | 1 | Drag-and-drop text move; **the range is encoded in the Activity string itself**, not in other columns |

### Reconstruction rules

`CursorPosition` is **post-edit**, which fixes every position:

- `Input` / `Paste`: insert at `CursorPosition - len(TextChange)`
- `Remove/Cut`: delete `len(TextChange)` chars starting at `CursorPosition`
  (holds for both `Backspace` and `Delete` — post-edit caret is the span start
  in both cases)
- `Replace`: split `TextChange` on the **last** `" => "`; the selection began at
  `CursorPosition - len(new)` and spanned `len(old)` chars
- `Nonproduction`: no text change; caret moves to `CursorPosition`

### Quirks

1. **Invalid UTF-8.** Some logs contain undecodable byte sequences. Read with
   `encoding="utf8-lossy"`; strict UTF-8 raises `ComputeError` and kills the scan.
2. **Backslash escapes in `TextChange`.** Newline is the two characters `\` `n`,
   not `0x0A`. Also `\"` and `\\`. Must be unescaped before use.
3. **Trailing newlines disagree.** Writers really do end essays with blank
   lines, and `texts/<writer>.txt` then adds one more `\n` on top. So neither
   side is authoritative about the tail: normalise with `rstrip("\n")` on
   *both* the replayed text and the gold text. Stripping only the gold side
   drops the exact-replay yield from 95.8% to 70.5%.
4. **Non-monotonic `DownTime`.** Key rollover means row *n+1* can have a smaller
   `DownTime` than row *n* (e.g. `l` at 48797 followed by `i` at 48959 while `l`
   is still held). Sort by `DownTime` before parsing, and expect overlaps.
5. **~4% of sessions do not replay exactly.** The adapter reproduces 575/600
   (95.8%) of WritingTask sessions byte for byte. The failures are
   transposition-shaped (`"cryptographcry"` for `"cryptography,"`), caused by
   the logger recording a stale `CursorPosition` during fast rollover. This is
   a corpus artifact, not a rule error — those sessions are **dropped**, not
   patched, so exact replay stays the gate.

---

## Aalto 136M Keystrokes

Source: `data/raw/aalto/`.

**Status: download incomplete at time of writing** (~1.0 GB of ~4.4 GB via
`curl` from `userinterfaces.aalto.fi/136Mkeystrokes/data/Keystrokes.zip`). The
archive has a valid local header but no end-of-central-directory record until
the transfer finishes, so it cannot be extracted or inspected yet.

This section must be filled in from the real files before `adapters/aalto.py`
is trusted — the same rule that applied to KLiCKe applies here. The column names
currently guessed in the plan (`PARTICIPANT_ID`, `TEST_SECTION_ID`, `SENTENCE`,
`PRESS_TIME`, `RELEASE_TIME`, `LETTER`) are **unverified**; the adapter's first
test asserts them against the real header and will fail loudly if they are wrong.
