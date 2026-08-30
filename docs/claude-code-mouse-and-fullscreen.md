# Clicking in the Claude Code CLI

Environment note, not a Typeshi finding. Recorded because the question
("can I patch the CLI so the mouse moves the caret?") has a surprising answer
and cost a session to establish: **the feature already exists, it is already
on, and there is nothing to patch.**

Verified 2026-08-13 against `2.1.231` on macOS 15.6 / Warp.

## The short version

Click-to-move-caret, drag-select, double-click-word and triple-click-line are
all shipped in Claude Code. Every one of them is gated on **fullscreen (alt
screen) mode**. Outside fullscreen the renderer never enables mouse tracking,
so the mouse does nothing and the feature is invisible.

`~/.claude/settings.json` already carries `"tui": "fullscreen"` on this
machine, so the gate is open. If a future session finds clicking dead, that
setting — or a stale process started before it was set — is the first thing
to check.

## Why "edit the CLI source" is not a path

The install here is the **native installer**, not npm:

    ~/.local/bin/claude -> ~/.local/share/claude/versions/2.1.231
    Mach-O 64-bit executable arm64, ~294 MB

That is a Bun-compiled single-file executable. There is no `cli.js` to edit.
The npm package is no longer the program either — `@anthropic-ai/claude-code`
is 7 files / 170 KB, a launcher that fetches the binary. And the binary
self-updates aggressively: `versions/` held five builds (2.1.224 → 2.1.231)
inside one week, so any patch would be overwritten within days.

The extension surface is settings, env vars and `~/.claude/keybindings.json`.
Treat the binary as read-only.

## How the two gates resolve

Minified identifiers below are build-specific and will drift; the *shape* is
the durable part.

**Mouse mode** — default is on, both env vars are opt-*outs*:

```js
function ome(){
  if (Q.CLAUDE_CODE_DISABLE_MOUSE       !== undefined) return Q.CLAUDE_CODE_DISABLE_MOUSE       ? "off"    : "full";
  if (Q.CLAUDE_CODE_DISABLE_MOUSE_CLICKS!== undefined) return Q.CLAUDE_CODE_DISABLE_MOUSE_CLICKS? "scroll" : "full";
  return "full";
}
```

**Fullscreen** — first match wins:

| precedence | source | effect |
| --- | --- | --- |
| 1 | `CLAUDE_CODE_SESSION_KIND=bg` | on |
| 2 | screen reader active | off |
| 3 | `CLAUDE_CODE_NO_FLICKER=0` / `DISABLE_ALTERNATE_SCREEN` | off |
| 4 | `CLAUDE_CODE_NO_FLICKER=1` | on (override) |
| 5 | tmux `-CC` control mode | off |
| 6 | Windows over SSH | off |
| 7 | `settings.json` → `"tui": "fullscreen"` \| `"default"` | on / off |
| 8 | rollout flags `tengu_amber_creek`, `tengu_pewter_brook` | on / off |

Row 8 is why this is invisible to some users and not others: with `tui`
unset, fullscreen is decided by a gradual rollout. Setting `tui` explicitly
takes it out of the experiment.

Everything selection-related then re-checks `altScreenActive` at dispatch
time (`dispatchClick`, `dispatchHover`, `handleSelectionDrag`,
`moveSelectionFocus` all bail early when it is false).

## What is actually implemented

- **Click → caret.** The prompt input carries `onClick`, which maps the cell
  to a text offset and sets the cursor:
  `measuredText.getOffsetFromPosition({line: ev.localRow + viewportStart, column: ev.localCol})`.
- **Drag.** The mouse dispatcher tests the SGR motion bit —
  `if ((btn & 32) !== 0) { onSelectionDrag(col,row); return }`. Selection
  state is `{anchor, focus, isDragging, anchorSpan, scope, …}`; press sets
  the anchor, motion moves the focus, release ends the drag.
- **Double / triple click.** A `clickCount` with both a debounce window and a
  distance tolerance dispatches `onMultiClick(col, row, 2|3)` for word and
  line select. Afterwards `anchorSpan.kind === "word"` makes further dragging
  extend by whole words.
- **Scoped selection.** The press records the element under it and columns
  are clamped to `scope.x1..x2`, so dragging inside the prompt box does not
  swallow the border or surrounding chrome.
- **Survives scrolling** via `virtualAnchorRow` / `scrolledOffAbove|Below`.
- **Copy** is `selection:copy`, bound to `cmd+c` and `ctrl+shift+c`, written
  through OSC 52 — so it works over SSH.
- **Keyboard selection**: `shift+arrows`, `shift+home|end` →
  `selection:extendLeft|Right|Up|Down|LineStart|LineEnd`.
- **Middle-click paste** on Linux/WSL/Windows; primary-selection paste on
  Linux.

The selection bindings live in the `Scroll` keybinding context and are
overridable in `~/.claude/keybindings.json`; the binary ships a JSON schema
listing every valid action name.

Warp specifically is a recognised terminal, not a fallback: the capability
layer has `macCmdClickArrivesWithoutSgrModifierBit()`, true for
`TERM_PROGRAM=WarpTerminal` on darwin, compensating for a Cmd-click quirk.

## Age

Not new, but not ancient, and the exact release could not be pinned:

- The click→caret handler is present in **all five** local builds
  (2.1.224 … 2.1.231), so it predates 2.1.224.
- The public changelog shows the mouse layer maturing earlier still —
  2.1.198 fixed "Cmd+click not opening URLs in fullscreen mode in Warp on
  macOS" and "double-click word selection in fullscreen mode"; 2.1.221 added
  "mouse-click support for multi-select menus … in fullscreen mode".

What is genuinely recent is fullscreen mode itself, which is the gate.

## The verification, and how to redo it

Static reading of a minified 294 MB binary produces confident nonsense, so
the claims above were driven end-to-end: fork a PTY, feed the output to a
`pyte` screen emulator, inject synthetic SGR mouse reports, read back the
resulting grid.

```
alt screen: True | mouse: True
found input at row=36 col=2: '❯ hello world'
TEST1 click->caret:   PASS  '❯ helXlo world'   (click at offset 3, then type X)
TEST2 drag-select:    PASS  5/5 cells highlighted, bg #264f78
TEST3 delete-select:        '❯ helXlo'          (backspace after drag)
```

The startup handshake, confirmed by capturing raw output:

    ESC[?1049h                                   alternate screen
    ESC[?1000h ESC[?1002h ESC[?1003h ESC[?1006h  press, drag, any-motion, SGR coords

SGR injection format is `ESC [ < btn ; col ; row M` to press, `m` to release,
`btn|32` for motion-while-pressed; **coordinates are 1-based**.

Four traps, all of which cost time:

1. **A PTY with no window size renders nothing.** `os.forkpty()` gives 0×0
   and the TUI never starts. Set `TIOCSWINSZ` immediately after the fork.
2. **Onboarding prompts block the TUI** (trust-this-folder, Chrome
   integration) and appear *before* alt screen is entered, so a naive capture
   sees no mouse sequences and looks like a negative result.
3. **Isolate the config.** Run with `CLAUDE_CONFIG_DIR=<scratch>` and a
   copied `.claude.json`, otherwise dismissing those prompts writes real
   settings — this investigation persisted `claudeInChromeDefaultEnabled:
   false` to `~/.claude.json` before switching to an isolated dir, and had to
   revert it.
4. **`grep` on the bundle yields false positives.** `boxSelectionEnabled` and
   `selectionMode` come from bundled cytoscape.js and AWS Bedrock schemas;
   `mouseDragged` comes from highlight.js keyword lists for Processing and
   Arduino. Filter to the JS region (offsets > ~250 MB here) and read the
   surrounding code before believing any hit.

Driver, trimmed to the load-bearing parts:

```python
import os, select, time, fcntl, termios, struct, pyte
COLS, ROWS = 120, 40
screen = pyte.Screen(COLS, ROWS); stream = pyte.ByteStream(screen)
env = dict(os.environ) | {"TERM": "xterm-256color",
                          "CLAUDE_CONFIG_DIR": os.path.abspath("cfg")}
pid, fd = os.forkpty()
if pid == 0:
    os.execve("/Users/peter/.local/bin/claude", ["claude"], env)
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
# ... pump(fd) into stream, locate text in screen.buffer, then:
os.write(fd, f"\x1b[<0;{col+1};{row+1}M".encode())   # press
os.write(fd, f"\x1b[<32;{col+4};{row+1}M".encode())  # drag
os.write(fd, f"\x1b[<0;{col+4};{row+1}m".encode())   # release
```

Cell highlight is readable as `screen.buffer[y][x].bg` (non-`"default"`) or
`.reverse`.

## If clicking ever stops working

In order: confirm `"tui": "fullscreen"` is still in `~/.claude/settings.json`;
restart the session, since fullscreen is decided once at startup; check that
neither `CLAUDE_CODE_DISABLE_MOUSE` nor `CLAUDE_CODE_DISABLE_MOUSE_CLICKS` has
appeared in the environment or a shell profile; hold Option while dragging if
the *terminal's* own selection is wanted instead of the app's; try
`/terminal-setup`.
