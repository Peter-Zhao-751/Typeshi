"""IteraTeR human edits -> synthetic composition sessions.

The corpus (docs/iterater-notes.md) carries real human revision passes as
edit actions with char offsets into the draft: R at [start, end) replaces,
A inserts, D deletes, ~7 per pass, with 145 documents forming real
D0 -> ... -> Dn chains. This is exactly the behaviour class the keystroke
corpora barely contain -- deliberate semantic revision -- and its absence
is why the model types cleanly but never rewrites (the IteraTeR gap,
docs/open-work.md).

Each session synthesized here TYPES the draft, then executes one or more
revision passes as CURSOR/SELDEL/KEY events, targeting the final revision.
Timing comes from a TimingSampler over real KLiCKe pools; a think pause
opens every action, a motor gap separates its ops. Windowing and per-window
labels are downstream's job (build_examples), and they are what keeps the
typing-heavy windows honestly labelled REV~0 while the revision windows
carry the high bins.

Two simplifications, both documented in the notes as open choices:
within-pass order is reading order (ascending offset, the dominant pattern
in real revision), and no typos are synthesized -- the windows are labelled
with their own true ECOR, so clean typing is accurate conditioning, not a
lie.

Every yielded session satisfies replay(events) == target byte-exactly; a
document whose offsets or chain do not verify is skipped and counted, never
emitted corrupted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from typeshi.buffer import TextBuffer
from typeshi.events import Event
from typeshi.serialize import _HOMOGLYPHS, normalize_typable

Session = tuple[str, str, list[Event]]


class _Mismatch(Exception):
    """An action's offsets or a chain's continuity failed verification."""


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """normalize_typable plus an index map: idx[i] is where input offset i
    landed in the output, so edit offsets survive length-changing entries
    (ellipsis, zero-widths, CRLF)."""
    out: list[str] = []
    idx: list[int] = []
    length = 0
    chars = list(text)
    for i, ch in enumerate(chars):
        idx.append(length)
        if ch == "\r" and i + 1 < len(chars) and chars[i + 1] == "\n":
            piece = ""  # the \n that follows survives on its own
        else:
            piece = _HOMOGLYPHS.get(ch, ch)
        out.append(piece)
        length += len(piece)
    idx.append(length)
    return "".join(out), idx


def _iter_rows(root: Path, sub: str) -> Iterator[dict]:
    for split in ("train", "dev", "test"):
        path = Path(root) / sub / f"{split}.json"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:  # JSON Lines despite the .json extension
                if line.strip():
                    yield json.loads(line)


def _pass_actions(row: dict) -> list[list[tuple]]:
    """One revision pass -> per-action step lists, offsets normalized and
    delta-adjusted to reading order. Raises _Mismatch when an action's
    `before` string is not what its span holds."""
    norm_before, idx = _normalize_with_map(row["before_revision"])
    actions = []
    delta = 0
    text = norm_before
    for a in sorted(row["edit_actions"],
                    key=lambda a: (a["start_char_pos"], a["end_char_pos"])):
        b = normalize_typable(a.get("before") or "")
        aft = normalize_typable(a.get("after") or "")
        s = idx[a["start_char_pos"]] + delta
        e = idx[a["end_char_pos"]] + delta
        if not (0 <= s <= e <= len(text)) or text[s:e] != b:
            raise _Mismatch(
                f"span [{s},{e}) holds {text[max(0, s):e]!r}, action says {b!r}"
            )
        if b == aft:
            continue
        steps: list[tuple] = []
        if e > s:
            steps += [("cursor", s), ("seldel", s, e)]
        elif aft:
            steps.append(("cursor", s))
        if aft:
            steps.append(("type", aft))
        actions.append(steps)
        text = text[:s] + aft + text[e:]
        delta += len(aft) - (e - s)
    return actions


# Executing a pass's actions faithfully does not always land byte-exact on
# after_revision: the spans often exclude the whitespace around a deleted
# word, leaving "aa  cc" where the revision says "aa cc" -- measured 303 of
# 559 docs, median residue ONE character. The residue is repaired with small
# synthesized cleanup edits (a human tidying a doubled space), because the
# byte-exact gate is non-negotiable; past this cap the row is genuinely
# inconsistent and inventing a large edit no human made would be fabricated
# behaviour, so the doc drops instead.
_RECONCILE_CAP = 32


def _reconcile(current: str, want: str) -> list[list[tuple]] | None:
    """Cleanup actions turning current into want, descending position so no
    offset bookkeeping is needed; None when the residue exceeds the cap."""
    import difflib

    ops = [o for o in difflib.SequenceMatcher(None, current, want).get_opcodes()
           if o[0] != "equal"]
    total = sum(max(i2 - i1, j2 - j1) for _, i1, i2, j1, j2 in ops)
    if total > _RECONCILE_CAP:
        return None
    actions: list[list[tuple]] = []
    for _, i1, i2, j1, j2 in reversed(ops):
        repl = want[j1:j2]
        steps: list[tuple] = []
        if i2 > i1:
            steps += [("cursor", i1), ("seldel", i1, i2)]
        elif repl:
            steps.append(("cursor", i1))
        if repl:
            steps.append(("type", repl))
        if steps:
            actions.append(steps)
    return actions


class _Synth:
    """Accumulates timed events against a live buffer."""

    def __init__(self, timing) -> None:
        self.timing = timing
        self.buf = TextBuffer()
        self.events: list[Event] = []
        self.t = 0

    def _at(self, gap: float) -> int:
        if self.events:
            self.t += max(1, int(round(gap)))
        return self.t

    def type_text(self, text: str) -> None:
        for ch in text:
            gap, hold = self.timing.key_timing(
                _pause_class_of(self.buf))
            press = self._at(gap)
            e = Event.key(ch, press, press + max(1, int(round(hold))))
            self.events.append(e)
            self.buf.apply(e)

    def run_action(self, steps: list[tuple]) -> None:
        first = True
        for step in steps:
            if step[0] == "type":
                self.type_text(step[1])
                continue
            gap = self.timing.think_pause() if first else self.timing.op_gap()
            first = False
            press = self._at(gap)
            e = (Event.cursor(step[1], press) if step[0] == "cursor"
                 else Event.seldel(step[1], step[2], press))
            self.events.append(e)
            self.buf.apply(e)


def _pause_class_of(buf: TextBuffer) -> str:
    from typeshi.adapters.timing import pause_class

    return pause_class(buf.text[: buf.cursor])


def iter_sessions(root: Path, timing, on_drop=None) -> Iterator[Session]:
    """(doc_id, final_text, events) per verified revision chain.

    Rows sharing a doc_id are chained where consecutive depths agree
    (this pass's after is the next pass's before); each maximal verified
    chain types its first draft, then executes every pass. A row that fails
    verification cuts its chain there -- the verified prefix is still
    emitted, the rest is dropped and reported via on_drop(doc_id, reason).
    """
    by_doc: dict[str, list[dict]] = {}
    for row in _iter_rows(root, "human_doc"):
        by_doc.setdefault(str(row["doc_id"]), []).append(row)

    for doc_id in sorted(by_doc):
        rows = sorted(by_doc[doc_id], key=lambda r: r["revision_depth"])
        chains: list[list[dict]] = []
        for row in rows:
            prev = chains[-1][-1] if chains and chains[-1] else None
            if (prev is not None
                    and row["revision_depth"] == prev["revision_depth"] + 1
                    and normalize_typable(row["before_revision"])
                    == normalize_typable(prev["after_revision"])):
                chains[-1].append(row)
            else:
                chains.append([row])
        for chain in chains:
            synth = _Synth(timing)
            try:
                synth.type_text(normalize_typable(chain[0]["before_revision"]))
                target = None
                for row in chain:
                    for steps in _pass_actions(row):
                        synth.run_action(steps)
                    target = normalize_typable(row["after_revision"])
                    if synth.buf.text != target:
                        cleanup = _reconcile(synth.buf.text, target)
                        if cleanup is None:
                            raise _Mismatch("residue past the reconcile cap")
                        for steps in cleanup:
                            synth.run_action(steps)
                    if synth.buf.text != target:
                        raise _Mismatch("buffer diverged from after_revision")
            except _Mismatch as exc:
                if on_drop is not None:
                    on_drop(doc_id, str(exc))
                continue
            if target is not None:
                yield doc_id, target, synth.events


def iter_sentence_sessions(root: Path, timing, on_drop=None) -> Iterator[Session]:
    """(doc_id, after_sent, events) dense mini-sessions from human_sent.

    Same annotations as the doc rows, at the scale of one sentence: type it,
    pause, revise it. These are the single-window, revision-dense examples;
    the writer id stays the doc_id so writer-hash splitting keeps every view
    of a document on the same side of the split.
    """
    for row in _iter_rows(root, "human_sent"):
        before = normalize_typable(row["before_sent"])
        after = normalize_typable(row["after_sent"])
        if before == after:
            continue
        prefix = 0
        limit = min(len(before), len(after))
        while prefix < limit and before[prefix] == after[prefix]:
            prefix += 1
        suffix = 0
        while (suffix < limit - prefix
               and before[-1 - suffix] == after[-1 - suffix]):
            suffix += 1
        s, e = prefix, len(before) - suffix
        replacement = after[prefix: len(after) - suffix]
        steps: list[tuple] = []
        if e > s:
            steps += [("cursor", s), ("seldel", s, e)]
        elif replacement:
            steps.append(("cursor", s))
        if replacement:
            steps.append(("type", replacement))
        synth = _Synth(timing)
        synth.type_text(before)
        synth.run_action(steps)
        if synth.buf.text != after:
            if on_drop is not None:
                on_drop(str(row["doc_id"]), "sentence diff did not verify")
            continue
        yield str(row["doc_id"]), after, synth.events
