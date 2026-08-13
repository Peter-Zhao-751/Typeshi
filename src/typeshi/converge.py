"""Convergence-guaranteed constrained decoding (design spec section 6).

Free-running composition generation composes its OWN essay: KLiCKe taught
"write like this", and nothing at sampling time binds the model to the given
target (probed: 2/5 exact starts, and unvalidated cursor moves crashed
replay). The guarantee cannot come from training; it comes from the mask.

This processor maintains the live TextBuffer during decoding and enforces:

  EVENT position:
    on-path (buffer is a prefix of the target):
        next-needed key + ANY key while excursion budget remains + BKSP
    off-path at the budget: BKSP only -- the excursion must resolve
  GAP position:
    every <DT:> always; EOS if and only if buffer == target

Timing is untouched -- every <DT:> and every hold bin stays model-sampled,
which is what keeps the realism numbers meaningful under the guarantee.

Stage 1 is linear: all editing is end-of-buffer, and with the cursor pinned
to the end "off-path depth" is the suffix beyond the common prefix.

Stage 2 re-admits <CUR:pos> and <SELDEL:a-b>. They are not vocabulary
entries -- they serialize as plain text through the base tokenizer -- so the
processor runs a digit-level state machine: entering an op commits the
sampler to one canonical token spelling (the tokenizer's own encoding of the
literal opener, then single-digit pieces, no leading zeros), and every digit
is masked so the growing number can still complete to a position valid for
the LIVE buffer (0 <= pos <= len for CUR; 0 <= a < b <= len for SELDEL).
The buffer applies the op the moment its '>' lands. Off-path depth
generalizes to edit distance against the best target prefix (the stage-1
suffix rule would count a whole shifted tail as off-path and ban exactly the
mid-buffer revisions stage 2 exists to admit), and completed ops join the
oscillation-guard discipline unchanged: ops open only while excursions are
open, a SELDEL that lands the buffer back on-path arms the cooldown, and one
that leaves depth at the budget forces resolution.

Compromises in stage 2, each chosen over unsound cleverness:
  - One canonical spelling per op. BPE tokenization is context-dependent
    ("12" may be one piece or two), so masking every token path that decodes
    to a valid op is infeasible; the mask instead forces the single spelling
    above. The decoded TEXT is unrestricted -- every valid position stays
    representable -- only the model's choice of pieces is narrowed.
  - Stage 2 needs tokenizer atoms: tok.encode, single-piece digits/'-'/'>'.
    A tokenizer that cannot supply them gets stage-1 behavior (ops masked
    out entirely) rather than a partially-checked op grammar.
  - Two states would deadlock resolution, and both force <CUR:len(buffer)>
    (cursor-to-end) as the only legal move: resolving with the cursor at 0
    (BKSP would no-op forever) and cooldown repayment with the cursor
    mid-buffer (the needed key only makes progress at the end). Jumping to
    the end to continue is also what real writers do after a revision; the
    restriction costs generality, never soundness.

Termination: resolution is BKSP while the cursor is off 0 (length strictly
shrinks; the empty buffer is on-path) and cursor-to-end when it is at 0, so
depth 0 is always reached; once on-path the cooldown rule steers the cursor
to the end, where the next-needed key is always sampleable. buffer == target
therefore stays reachable from every state; max_new_tokens bounds the walk,
and a generation that never converges is a FAILED attempt for the caller to
count, never a silently wrong text.
"""

from __future__ import annotations

from typeshi.buffer import TextBuffer
from typeshi.serialize import _encode_char, special_tokens


def _prefix_edit_depth(text: str, target: str) -> int:
    """Edits needed to land `text` on some prefix of `target`.

    min over prefixes of Levenshtein(text, prefix): the number of one-char
    repairs separating the buffer from being on-path. Coincides with the
    stage-1 suffix rule whenever the divergence is a pure appended tail.
    """
    m = len(text)
    row = list(range(m + 1))
    best = m
    for tc in target:
        prev = row
        row = [prev[0] + 1]
        for i in range(1, m + 1):
            row.append(min(prev[i] + 1, row[i - 1] + 1,
                           prev[i - 1] + (text[i - 1] != tc)))
        best = min(best, row[m])
        if best == 0:
            return 0
    return best


def _completable(w: int, lo: int, hi: int) -> bool:
    """Can appending zero or more digits to `w` land it inside [lo, hi]?

    Appending k digits to w spans exactly [w*10^k, w*10^k + 10^k - 1]; the
    loop checks each span for intersection until it overshoots hi.
    """
    span = 1
    while w * span <= hi:
        if w * span + span - 1 >= lo:
            return True
        span *= 10
    return False


def _digit_moves(digits: str, lo: int, hi: int) -> tuple[list[str], bool]:
    """Legal continuations of a partial number constrained to [lo, hi].

    Returns (next digits, whether the field may close now). Canonical
    spellings only -- no leading zeros -- so every position has exactly one
    token path and the mask never has to reason about "007".
    """
    can_close = bool(digits) and lo <= int(digits) <= hi
    if digits == "0":
        return [], can_close
    out = []
    for d in "0123456789":
        if not digits and d == "0":
            if lo == 0:
                out.append(d)
            continue
        if _completable(int(digits + d), lo, hi):
            out.append(d)
    return out, can_close


class ConvergenceProcessor:
    """LogitsProcessor guaranteeing the stream types `target` exactly.

    Batch size 1, matching generate(). `excursion_budget` is the maximum
    off-path depth in characters before the mask offers only resolution.
    Cursor ops (stage 2) activate only when the tokenizer supplies the
    plain-text atoms; otherwise the processor is exactly the stage-1 mask.
    """

    def __init__(self, tok, prompt_len: int, target: str,
                 excursion_budget: int = 4, resolve_progress: int = 2,
                 written_so_far: str = "", cursor: int | None = None) -> None:
        """`written_so_far`/`cursor` seed the buffer for a CONTINUATION
        window (windowed generation, matching how composition was trained:
        512-event windows with a <WRITTEN> tail). A continuation starts in
        GAP slot so the model emits the window-boundary <DT:> itself --
        format v2 allows a leading gap on continuation windows, and
        fabricating that gap host-side would be invented timing."""
        import torch

        self.prompt_len = prompt_len
        self.target = target
        self.budget = excursion_budget
        # Oscillation guard, added after the first live probe: the phase-2
        # model WANTS its own essay, so with excursions always open it typed
        # wrong -> was forced back -> typed wrong again, burning the whole
        # token budget at ~50% BKSP (1/5 converged). Once a forced
        # resolution completes, excursions stay closed until
        # `resolve_progress` on-path characters are typed: every cycle nets
        # progress, so stage-1 convergence is bounded, adversarial samplers
        # included. Stage-2 CUR hops neither edit text nor arm the cooldown,
        # so a sampler that only moves the caret converges never -- there the
        # bound is max_new_tokens and the guarantee stays terminated-implies-
        # exact (module docstring, Termination).
        self.resolve_progress = resolve_progress
        self._resolving = False
        self._cooldown = 0
        self.buffer = TextBuffer(written_so_far, cursor)
        self._consumed = 0  # generated tokens already applied to the buffer
        self._slot = "gap" if written_so_far else "event"
        self._op: dict | None = None  # in-flight <CUR:/<SELDEL: state
        self._depth_key: str | None = None
        self._depth_val = 0

        # id <-> role tables, built once from the registered vocabulary.
        self._char_ids: dict[str, list[int]] = {}
        dt_ids, bksp_ids = [], []
        self._id_char: dict[int, str] = {}
        self._id_kind: dict[int, str] = {}
        for t in special_tokens():
            if not t.endswith(">"):
                continue
            i = tok.convert_tokens_to_ids(t)
            if i is None or i == tok.unk_token_id:
                continue
            if t.startswith("<DT:"):
                dt_ids.append(i)
                self._id_kind[i] = "dt"
            elif t.startswith("<BKSP:"):
                bksp_ids.append(i)
                self._id_kind[i] = "bksp"
            elif ":" in t and not t.startswith(
                ("<MODE:", "<WPM:", "<ECOR:", "<EUNC:", "<REV:")
            ):
                name = t[1:].rsplit(":", 1)[0]
                from typeshi.serialize import _decode_char

                char = _decode_char(name)
                if len(char) == 1:
                    self._char_ids.setdefault(char, []).append(i)
                    self._id_char[i] = char
                    self._id_kind[i] = "key"

        missing = {c for c in target if c not in self._char_ids}
        if missing:
            raise ValueError(f"target contains unsupported chars: {missing!r}")

        self._dt = torch.tensor(dt_ids)
        self._bksp = torch.tensor(bksp_ids)
        self._all_keys = torch.tensor(
            [i for ids in self._char_ids.values() for i in ids]
        )
        eos = tok.eos_token_id
        self._eos = torch.tensor([eos] if eos is not None else [])
        self._per_char = {c: torch.tensor(ids) for c, ids in self._char_ids.items()}
        self._dev: dict[str, object] = {}

        self._ops = self._build_op_tables(tok)
        if self._ops is not None:
            root = self._ops["root"]["children"]
            self._op_root_all = torch.tensor(list(root))
            self._op_root_cur = torch.tensor(
                [i for i, node in root.items() if "cur" in node["kinds"]]
            )

    def _build_op_tables(self, tok) -> dict | None:
        """Token atoms for the <CUR:>/<SELDEL:> plain-text spellings.

        Openers are the tokenizer's own encodings of the literal prefixes
        (multi-piece is fine; a trie walks them). Digits, '-' and '>' must
        each be a single piece so the position machine advances one digit
        per step. Any atom missing, unknown, or colliding with a grammar
        token disables stage 2 -- ops stay masked out, exactly stage 1.
        """
        if not hasattr(tok, "encode"):
            return None

        def ids(s: str) -> list[int]:
            return [int(i) for i in tok.encode(s, add_special_tokens=False)]

        unk = tok.unk_token_id
        singles: dict[str, int] = {}
        for ch in "0123456789->":
            seq = ids(ch)
            if len(seq) != 1 or seq[0] == unk or seq[0] in self._id_kind:
                return None
            singles[ch] = seq[0]
        root: dict = {"children": {}, "kinds": set(), "leaf": None}
        for text, kind in (("<CUR:", "cur"), ("<SELDEL:", "seldel")):
            seq = ids(text)
            if not seq or unk in seq or any(i in self._id_kind for i in seq):
                return None
            root["kinds"].add(kind)
            node = root
            for i in seq:
                node = node["children"].setdefault(
                    i, {"children": {}, "kinds": set(), "leaf": None}
                )
                node["kinds"].add(kind)
            node["leaf"] = kind
        return {"singles": singles,
                "chars": {v: k for k, v in singles.items()},
                "root": root}

    def _to(self, name: str, tensor, device):
        key = (name, device)
        if key not in self._dev:
            self._dev[key] = tensor.to(device)
        return self._dev[key]

    def _depth(self) -> int:
        text = self.buffer.text
        if text != self._depth_key:
            self._depth_key = text
            self._depth_val = _prefix_edit_depth(text, self.target)
        return self._depth_val

    def _forced_cur(self) -> bool:
        """States whose only sound move is cursor-to-end (<CUR:len>).

        Resolving with the cursor at 0: BKSP would no-op forever. Cooldown
        repayment with the cursor mid-buffer: the needed key extends the
        text only at its end. Both are unreachable without cursor ops, so
        stage-1 tokenizers never see this branch.
        """
        if self._ops is None:
            return False
        cursor, n = self.buffer.cursor, len(self.buffer.text)
        depth = self._depth()
        return (self._resolving and depth > 0 and cursor == 0) or (
            self._cooldown > 0 and depth == 0 and cursor < n
        )

    def _feasible_kinds(self, forced: bool) -> set[str]:
        # SELDEL needs a nonempty range to delete; CUR:0 is always valid.
        if forced:
            return {"cur"}
        kinds = {"cur"}
        if self.buffer.text:
            kinds.add("seldel")
        return kinds

    def _op_bounds(self, op: dict) -> tuple[int, int]:
        n = len(self.buffer.text)  # static for the whole op
        if op["kind"] == "cur":
            return (n, n) if op["forced"] else (0, n)
        if op["phase"] == "a":
            return 0, n - 1  # a < b <= n needs a <= n-1
        return op["a"] + 1, n

    def _start_op(self, node: dict, forced: bool) -> None:
        if node["leaf"]:
            self._op = {"kind": node["leaf"], "phase": "a", "digits": "",
                        "a": None, "forced": forced}
        else:
            self._op = {"node": node, "forced": forced}

    def _apply_op_token(self, i: int) -> None:
        op = self._op
        assert op is not None
        if "node" in op:
            node = op["node"]["children"][i]
            if node["leaf"]:
                self._start_op(node, op["forced"])
            else:
                op["node"] = node
            return
        ch = self._ops["chars"][i]
        if ch == "-":
            op["a"] = int(op["digits"])
            op["phase"] = "b"
            op["digits"] = ""
        elif ch == ">":
            self._finish_op(op)
        else:
            op["digits"] += ch

    def _finish_op(self, op: dict) -> None:
        """Applies a completed op and folds it into the excursion guard.

        CUR moves the caret only: depth is unchanged, so the guard is too
        (the excursion, if any, is the edit that follows). SELDEL edits
        text and gets the BKSP rules verbatim: landing on-path arms the
        cooldown, leaving depth at the budget forces resolution.
        """
        if op["kind"] == "cur":
            self.buffer._move(int(op["digits"]))
        else:
            self.buffer._seldel(op["a"], int(op["digits"]))
            if self._depth() == 0:
                self._resolving = False
                self._cooldown = self.resolve_progress
            elif self._depth() >= self.budget:
                self._resolving = True
        self._op = None
        self._slot = "gap"

    def _apply_new_tokens(self, ids: list[int]) -> None:
        """Replays committed tokens and advances the guard state.

        State lives HERE, derived from the token stream, not in __call__ --
        the mask for a position must depend only on what was emitted before
        it, never on how many times the processor happened to be invoked.

        The cooldown arms on ANY backspace that lands on-path, not just
        budget-forced ones: shallow type-one-wrong/delete loops below the
        budget are otherwise free to oscillate forever, and so is deleting
        correct text and retyping it. A BKSP that deleted nothing (cursor
        at 0) advances nothing.
        """
        for i in ids:
            if self._op is not None:
                self._apply_op_token(i)
                continue
            kind = self._id_kind.get(i)
            if kind == "key":
                self.buffer._insert(self._id_char[i])
                if self._depth() == 0 and self._cooldown > 0:
                    self._cooldown -= 1
                elif self._depth() >= self.budget:
                    self._resolving = True
                self._slot = "gap"
            elif kind == "bksp":
                deleted = self.buffer.cursor > 0
                self.buffer._backspace()
                if deleted and self._depth() == 0:
                    self._resolving = False
                    self._cooldown = self.resolve_progress
                self._slot = "gap"
            elif kind == "dt":
                self._slot = "event"
            elif self._ops is not None and i in self._ops["root"]["children"]:
                self._start_op(self._ops["root"]["children"][i],
                               forced=self._forced_cur())
            # anything else: no buffer effect

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids.shape[1] - self.prompt_len
        new = input_ids[0, self.prompt_len + self._consumed:].tolist()
        self._apply_new_tokens(new)
        self._consumed = generated

        mask = torch.full_like(scores, float("-inf"))
        device = scores.device

        if self._op is not None:
            # Mid-op: only continuations that can still complete validly.
            op = self._op
            if "node" in op:
                kinds = self._feasible_kinds(op["forced"])
                ids = [i for i, node in op["node"]["children"].items()
                       if node["kinds"] & kinds]
            else:
                lo, hi = self._op_bounds(op)
                digits, can_close = _digit_moves(op["digits"], lo, hi)
                singles = self._ops["singles"]
                ids = [singles[d] for d in digits]
                if can_close:
                    closer = ("-" if op["kind"] == "seldel"
                              and op["phase"] == "a" else ">")
                    ids.append(singles[closer])
            mask[:, torch.tensor(ids, device=device)] = 0
            return scores + mask

        if self._slot == "gap":
            # GAP position: timing is free; ending is earned.
            allowed = [self._to("dt", self._dt, device)]
            if self.buffer.text == self.target and self._eos.numel():
                allowed.append(self._to("eos", self._eos, device))
        else:
            # EVENT position. All guard state was advanced token-by-token in
            # _apply_new_tokens; this branch only reads it.
            text = self.buffer.text
            cursor = self.buffer.cursor
            n = len(text)
            depth = self._depth()

            if self._forced_cur():
                allowed = [self._to("op_cur", self._op_root_cur, device)]
            else:
                allowed = []
                if depth == 0 and cursor == n and n < len(self.target):
                    needed = self.target[n]
                    allowed.append(
                        self._to(f"c:{needed}", self._per_char[needed], device)
                    )
                excursions_open = (
                    depth < self.budget
                    and not self._resolving
                    and self._cooldown == 0
                )
                if excursions_open:
                    allowed.append(self._to("keys", self._all_keys, device))
                    if self._ops is not None:
                        roots = (self._op_root_all if text
                                 else self._op_root_cur)
                        allowed.append(
                            self._to(f"op_root:{bool(text)}", roots, device)
                        )
                if text and cursor > 0 and not (self._cooldown > 0 and depth == 0):
                    # No BKSP while repaying progress on-path: deleting
                    # correct text there would reopen the oscillation loop.
                    allowed.append(self._to("bksp", self._bksp, device))
                if not allowed:
                    # Corner states must still offer something legal: type
                    # the next needed char if the target is unfinished, else
                    # undo one char (retype cycle) -- never an off-path
                    # extension.
                    if n < len(self.target):
                        needed = self.target[n]
                        allowed.append(
                            self._to(f"c:{needed}", self._per_char[needed], device)
                        )
                    else:
                        allowed.append(self._to("bksp", self._bksp, device))

        for ids in allowed:
            mask[:, ids] = 0
        return scores + mask
