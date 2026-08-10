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
| Aalto 136M Keystrokes | `userinterfaces.aalto.fi/136Mkeystrokes/` | **Verified** — `Keystrokes/files/readme.txt` grants use "for non-commercial use in your own research or projects with attribution to the authors" |
| KLiCKe | Google Drive link from `github.com/terryyutian/KLiCKe-Corpus` | No terms file in the archive |

**Aalto:** the readme's grant covers research model training and requires
citing Dhakal, Feit, Kristensson & Oulasvirta, *Observations on Typing from
136 Million Keystrokes*, CHI 2018 (doi 10.1145/3173574.3174220). It is
**non-commercial only** — a commercial deployment of a model trained on this
data is not covered.

**Open item for the user:** KLiCKe ships no terms file. Confirm its
distribution terms permit model training before any checkpoint trained on it
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
5. **Paste and drag-drop sessions are dropped (13.8% + 1%).** A `Paste` row
   expands to one KEY event per character at a single timestamp — 142 paste
   rows became 15,031 zero-IKI "keystrokes" in 500 logs. A motor model must
   not learn that. Phase 2 may add a real `<PASTE>` event instead.
6. **`Nonproduction` rows carry the caret's real move time.** The adapter
   stamps a cursor move with the navigation row's own timestamp when the next
   edit confirms the position (85,970 affected rows in 500 logs); otherwise a
   pause between moving and typing lands on the wrong side of the move.
7. **~4% of sessions do not replay exactly.** The adapter reproduces 575/600
   (95.8%) of WritingTask sessions byte for byte. The failures are
   transposition-shaped (`"cryptographcry"` for `"cryptography,"`), caused by
   the logger recording a stale `CursorPosition` during fast rollover. This is
   a corpus artifact, not a rule error — those sessions are **dropped**, not
   patched, so exact replay stays the gate.

---

## Aalto 136M Keystrokes

Source: `data/raw/aalto/`.

Downloaded (1.57 GB) and extracted to 18 GB / 168,595 files.

### Layout

| Path | Contents |
|---|---|
| `Keystrokes/files/<participant>_keystrokes.txt` | 168,593 logs, one per participant, 15 sentences each |
| `Keystrokes/files/metadata_participants.txt` | Demographics + aggregate speed/error stats |
| `Keystrokes/files/readme.txt` | Official column documentation and license |

**Files are TAB-separated with a `.txt` extension**, not CSV. The test fixture
keeps the `.txt` extension for that reason.

### Columns

One row = one keypress. Column names match the plan's guesses exactly.

| Column | Dtype | Meaning |
|---|---|---|
| `PARTICIPANT_ID` | Int64 | Participant |
| `TEST_SECTION_ID` | Int64 | The presented sentence — **this is the session key** |
| `SENTENCE` | String | Target text shown to the participant |
| `USER_INPUT` | String | What the participant actually submitted — tighter replay ground truth than `SENTENCE` |
| `KEYSTROKE_ID` | Int64 | Keypress ID |
| `PRESS_TIME` / `RELEASE_TIME` | Int64 | **Unix epoch ms**, so sessions must be rebased to zero |
| `LETTER` | String | Literal character, or a name: `BKSP`, `SHIFT`, `CAPS_LOCK` |
| `KEYCODE` | Int64 | JavaScript keycode (8 = backspace, 16 = shift, 32 = space) |

Useful `metadata_participants.txt` fields: `KEYBOARD_TYPE`
(`full` 73,759 / `laptop` 91,250 / `small` 1,886 / `on-screen` 1,699) and
`NATIVE_LANGUAGE` (`en` 141,499 of 168,594).

### Quirks

1. **Quote characters shear rows.** Sentences contain apostrophes and quotation
   marks. Read with `quote_char=None`, otherwise polars treats them as field
   delimiters and emits `CSV malformed` warnings while silently mangling rows.
2. **`LETTER` is null on ~7.8% of rows**, concentrated in about 10% of
   participants — the readme's "keystrokes are not logged or not displayed
   correctly, the keycode is used instead" case. A keycode cannot recover
   letter *case*, so affected sessions are **dropped** rather than guessed at.
3. **`LETTER` can disagree with `KEYCODE`.** Participant 100001 logs
   `LETTER='y'` on a row whose `KEYCODE=84` (`T`). Same root cause as above.
4. **Modifier rows carry no text.** `SHIFT` and `CAPS_LOCK` rows are skipped;
   only single characters and `BKSP` become events.
5. **Replay fidelity against `USER_INPUT`:** over 1,605 real sessions, 76.0%
   replay byte-exact, mean edit-distance similarity 0.9875, and 98.0% score
   at least 0.90. The residual is corpus logging noise, not adapter error.
   Compare with edit distance, never positional overlap — one dropped
   keystroke shifts every later character.
6. **Mobile is out of scope** per the plan. Filter to `KEYBOARD_TYPE` in
   {`full`, `laptop`} via `aalto.physical_keyboard_participants()`.
