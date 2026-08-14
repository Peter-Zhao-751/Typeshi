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

---

## How We Type (DRAFT — schema read off the bytes 2026-08-13, adapter new)

Source: `data/raw/howwetype/`. Downloaded from Zenodo record 4034268
(`zenodo.org/record/4034268`), linked from `userinterfaces.aalto.fi/how-we-type/`.
Only the small files were taken — `Typing.zip` (733 kB), `Readme.txt`,
`keyboard_flat_coordinates.csv`, `Background.xlsx`. The mocap / eye tracking /
video archives (≈300 GB total) were **not** downloaded.

**License: CC BY-NC 4.0** (declared on the Zenodo record). The bundled
`Readme.txt` grants use "for non-commercial use in your own research with
attribution" and requires citing Feit, Weir & Oulasvirta, *How We Type:
Movement Strategies and Performance in Everyday Typing*, CHI 2016
(doi 10.1145/2858036.2858233). Non-commercial only, same caveat as Aalto 136M.

**Why this corpus:** every keypress is annotated with the finger that pressed
it (from motion capture), which turns same-finger detection into supervised
ground truth for the keyboard-reconstruction workstream.

### Layout

| Path | Contents |
|---|---|
| `Typing/<user>_log_Sentences_<epoch>_matched.txt` | 30 logs, one per participant, 49–50 sentence blocks each, 36,955 keypress rows total |
| `Typing/Background.xlsx`, `Background.xlsx` | Post-study survey (duplicated inside and outside the zip) |
| `keyboard_flat_coordinates.csv` | Per-key x/y/z of key centers, Escape at origin; Swedish/Finnish physical layout |
| `Readme.txt` | Official column documentation and license |

This release contains only the `sentences` condition; the paper's `random` and
`mix` conditions are not included (per the readme).

### Typing log columns

**TAB-separated with a `.txt` extension.** One row = one keypress (down only —
**there are no key-release times anywhere in this corpus**). Actual header,
which disagrees with the readme (extra `wpm`, `sd_iki`; `finger` values are
`L_Index`-style, not the readme's `Hands_L_L4` marker names):

| Column | Dtype | Meaning |
|---|---|---|
| `` (unnamed) | Int64 | 0-based row counter |
| `input_time` | Float64 | Press time, **Unix epoch seconds** with ms decimals — needs ×1000 and rebasing |
| `user_id` | Int64 | Participant (4–6 digit number; also in the filename) |
| `stimulus_id` | Int64 | The presented sentence — **this is the session key**, like Aalto's `TEST_SECTION_ID` |
| `input_index` | Int64 | Row index within the stimulus block (counts modifier rows too) |
| `iki` | Int64 | Inter-key interval, ms; 0 on the block's first row; ≈ press-time delta ±1 ms rounding |
| `input` | String | The rendered output: literal char, `_` for space, `\x08` for backspace, modifier names, or **multi-char dead-key artifacts** (`` `? ``, `´´`, `''`) |
| `key_symbol` | String | X11 keysym: lowercase letters, `space`, `BackSpace`, `Shift_L`/`Shift_R`, `period`, `question`, `adiaeresis` (ä), `odiaeresis` (ö), `aring` (å), `Multi_key`, … |
| `current_input` | String | The submitted response for this block, **constant across the block** (filled in retroactively) — the replay ground truth, like Aalto's `USER_INPUT` |
| `wpm` / `sd_iki` | Float64 | Per-block aggregate stats, constant across the block |
| `finger` | String | **The finger annotation**: `{L,R}_{Thumb,Index,Middle,Ring,Little}`, zero nulls in the release |
| `right_hand` | Int64 | 1 if the right hand pressed the key |
| `stimulus` | String | Target sentence shown (mostly Finnish, some English); constant across the block |
| `bigram` / `last_input` / `last_finger` | String | Previous-key context, `*`/`NaN` on block-initial rows — derived, ignored by the adapter |

### Quirks

1. **No key releases.** Only key-down times exist. The adapter sets
   `release_time = press_time`; hold-time statistics cannot come from this
   corpus.
2. **Letters are logged lowercase even when shifted.** `Shift_L` + `s`
   producing "S" logs `key_symbol='s'`, `input='s'`. Shifted *symbols* arrive
   already resolved (`question` → `?`, `exclam` → `!`), and shifted Nordic
   letters occasionally arrive as capital keysyms (`Aring` → `Å`). Case is
   reconstructed by capitalizing the first character-producing keypress after
   a Shift row; verified by replay, 98.1% of blocks reproduce `current_input`
   byte-exact and 99.87% score ≥ 0.90 edit similarity (the 2 failures are
   rollover-corrupted logs, same shape as the other corpora — dropped by the
   gate).
3. **Read WITH quote handling — the opposite of Aalto.** These files were
   written with CSV quoting: fields containing `"` are wrapped in quotes with
   internal quotes doubled (`""""` = one literal `"`). Reading with
   `quote_char=None` leaves those artifacts in `input`/`current_input` and
   costs replay exactness; the default quote char parses them cleanly and row
   counts are identical either way.
4. **No Enter/Return rows.** The block just ends; `stimulus_id` increments on
   the next row.
5. **Dead-key artifacts.** 10 rows have multi-char `input` (`` `? ``, `´´`,
   `´k`, `''`, `´\x08`) from the acute/compose dead keys on the Finnish
   layout. The adapter emits one KEY event per rendered char at the same
   timestamp; blocks the artifacts corrupt fail the replay gate and are
   dropped.
6. **One file has a phantom `Unnamed: 10` column** (549687). Select columns by
   name, never by position.
7. **Timestamps are coarse.** Keypresses were polled at 40 ms (readme), so
   IKIs are quantized — do not mix this corpus into hold-time or
   fine-IKI distribution training without remembering that.
8. **One block is non-monotonic in `input_time`** (rollover, as in the other
   corpora). Sort by `input_time` before parsing.
9. **`keyboard_flat_coordinates.csv` is TAB-separated despite the name**, 62
   rows, and the key `0` appears twice with different coordinates (top-row vs
   numpad, most likely). Coordinates are meters-ish normalized with Escape at
   (0,0,0).
10. **`stimulus` ≠ `current_input` in 18% of blocks** — participants made
   uncorrected errors. Gate replay against `current_input`, not `stimulus`.
