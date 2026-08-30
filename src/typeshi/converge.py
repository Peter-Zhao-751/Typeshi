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
mid-buffer revisions stage 2 exists to admit). A SELDEL that lands the buffer
back on-path arms the cooldown; one that leaves the excursion unaffordable
forces resolution.

Stage 3 makes semantic revision representable. Three coupled changes:

  - CUR is ungated on-path. A caret move edits no text, so it cannot break
    convergence, and sharing the typo gate made every mid-buffer revision
    unreachable while typing normally. SELDEL stays gated -- it deletes.
  - The excursion budget becomes an AFFORDABILITY check rather than a
    constant. An excursion of any length is permitted while the remaining
    token budget can still pay to undo it and finish the target. A constant
    is wrong at both ends: 4 forbids a 10-50 character semantic revision
    outright, and a large one lets a wandering sampler burn the allowance.
    SELDEL is what makes this practical -- undoing 50 characters costs one
    caret move plus one delete, not 50 backspaces -- and the leash tightens
    by itself as the budget depletes, so the model revises early and
    polishes late.
  - STALENESS forces prompt repair. Edit distance does not grow as you type
    past a divergence, so a typo at character 3 still reads as depth 1 forty
    characters later and nothing forces a fix until EOS is blocked at the
    very end -- whereupon the only repair is a backspace run as long as
    everything typed since. Events-since-on-path is what expresses "this has
    been wrong for a while".

The price quoted by the affordability check is the cheapest route the mask
can COMPEL, never the cheapest the model might choose: authorising a long
excursion because CUR+SELDEL could undo it in two events, then letting the
model backspace fifty times instead, would strand the run with no budget and
a wrong buffer. At the floor the mask therefore forces that route.

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
shrinks; the empty buffer is on-path), SELDEL (which also strictly shrinks),
and cursor-to-end when the caret is at 0, so depth 0 is always reached; once
on-path the cooldown rule steers the cursor to the end, where the
next-needed key is always sampleable. Caret moves edit nothing and so cannot
pay down staleness or the budget; the hop cap bounds any caret-only walk,
which is what keeps an ungated CUR from replacing the old max_new_tokens
argument with an infinite loop. buffer == target therefore stays reachable
from every state; max_new_tokens bounds the walk, and a generation that
never converges is a FAILED attempt for the caller to count, never a
silently wrong text.
"""

from __future__ import annotations

from typeshi.buffer import TextBuffer
from typeshi.serialize import _encode_char, special_tokens


def _common_prefix_len(a: str, b: str) -> int:
    """Index where the two strings first differ.

    The divergence point, which is where a caret must land for one SELDEL to
    excise everything wrong -- distinct from _prefix_edit_depth, which says
    how MANY edits separate the buffer from any prefix, not WHERE.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


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


def role_tables(tok) -> tuple[dict[str, list[int]], dict[int, str], dict[int, str]]:
    """Maps the registered grammar vocabulary onto decoding roles.

    Returns (char -> ids, id -> char, id -> kind) where kind is one of
    "key" / "bksp" / "dt". Shared with the streaming observer in
    typeshi.generate, which has to replay committed tokens into a buffer
    exactly the way this processor does -- two copies of this table drifting
    apart would make the live view disagree with the final event list.
    """
    from typeshi.serialize import _decode_char

    char_ids: dict[str, list[int]] = {}
    id_char: dict[int, str] = {}
    id_kind: dict[int, str] = {}
    for t in special_tokens():
        if not t.endswith(">"):
            continue
        i = tok.convert_tokens_to_ids(t)
        if i is None or i == tok.unk_token_id:
            continue
        if t.startswith("<DT:"):
            id_kind[i] = "dt"
        elif t.startswith("<BKSP:"):
            id_kind[i] = "bksp"
        elif ":" in t and not t.startswith(
            ("<MODE:", "<WPM:", "<ECOR:", "<EUNC:", "<REV:")
        ):
            name = t[1:].rsplit(":", 1)[0]
            char = _decode_char(name)
            if len(char) == 1:
                char_ids.setdefault(char, []).append(i)
                id_char[i] = char
                id_kind[i] = "key"
    return char_ids, id_char, id_kind


class ConvergenceProcessor:
    """LogitsProcessor guaranteeing the stream types `target` exactly.

    Batch size 1, matching generate(). `excursion_budget` is the maximum
    off-path depth in characters before the mask offers only resolution.
    Cursor ops (stage 2) activate only when the tokenizer supplies the
    plain-text atoms; otherwise the processor is exactly the stage-1 mask.
    """

    def __init__(self, tok, prompt_len: int, target: str,
                 excursion_budget: int = 4, resolve_progress: int = 2,
                 written_so_far: str = "", cursor: int | None = None,
                 token_budget: int | None = None,
                 staleness_window: int = 10,
                 caret_hop_cap: int = 3) -> None:
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
        # Affordability (spec: excursions of any length, provided the budget
        # can still pay to undo them and finish) plus the two bounds that make
        # a wide budget safe: staleness forces a stale divergence to be
        # repaired promptly, and the hop cap stops a caret-only walk.
        self._token_budget = token_budget
        self.staleness_window = staleness_window
        self.caret_hop_cap = caret_hop_cap
        self._stale = 0      # events emitted since the buffer was on-path
        self._hops = 0       # caret moves since the correct prefix last grew
        self._prefix = 0     # high-water mark of the correct prefix
        self._generated = 0  # tokens emitted so far, for the budget check
        self.buffer = TextBuffer(written_so_far, cursor)
        self._consumed = 0  # generated tokens already applied to the buffer
        self._slot = "gap" if written_so_far else "event"
        self._op: dict | None = None  # in-flight <CUR:/<SELDEL: state
        self._depth_key: str | None = None
        self._depth_val = 0

        # id <-> role tables, built once from the registered vocabulary.
        self._char_ids, self._id_char, self._id_kind = role_tables(tok)
        dt_ids = [i for i, k in self._id_kind.items() if k == "dt"]
        bksp_ids = [i for i, k in self._id_kind.items() if k == "bksp"]

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
        self._op_len: dict[str, int] = {}
        if self._ops is not None:
            self._op_len = {
                "cur": len(tok.encode("<CUR:", add_special_tokens=False)),
                "seldel": len(tok.encode("<SELDEL:", add_special_tokens=False)),
            }
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

    # -- cost model -------------------------------------------------------
    #
    # The affordability rule is only as sound as the price it quotes. It must
    # be the cost of the cheapest route the mask can COMPEL, never the
    # cheapest route the model might choose: authorising a 50-character
    # excursion because CUR+SELDEL could undo it in two events, and then
    # letting the model backspace fifty times instead, strands the run with
    # no budget and a wrong buffer. Everything below is measured in TOKENS,
    # because that is what max_new_tokens actually bounds -- an event is two
    # tokens (the event and its gap), but an op costs a digit per decimal
    # place and so grows with buffer position.

    MARGIN_TOKENS = 8

    # Extra staleness allowance per character of depth (see _stale_forced).
    STALE_DEPTH_SLACK = 2

    @staticmethod
    def _digits(n: int) -> int:
        return len(str(n))

    def _cur_tokens(self, pos: int) -> int:
        return self._op_len["cur"] + self._digits(pos) + 2  # '>' and the gap

    def _seldel_tokens(self, a: int, b: int) -> int:
        return (self._op_len["seldel"] + self._digits(a) + 1
                + self._digits(b) + 2)  # '-', '>' and the gap

    def _finish_tokens(self, text: str, cursor: int) -> float:
        """Tokens to reach `text == target` by the cheapest compellable route."""
        if text == self.target:
            return 1  # just the EOS
        p = _common_prefix_len(text, self.target)
        cost = 0
        if len(text) > p:
            if self._ops is not None:
                if cursor != p:
                    cost += self._cur_tokens(p)
                cost += self._seldel_tokens(p, len(text))
            elif cursor == len(text):
                cost += 2 * (len(text) - p)  # stage 1: backspace the tail
            else:
                return float("inf")  # no way to move the caret, no way back
        elif cursor != len(text):
            if self._ops is None:
                return float("inf")
            cost += self._cur_tokens(len(text))
        return cost + 2 * (len(self.target) - p) + 1

    def _remaining(self) -> float:
        if self._token_budget is None:
            return float("inf")
        return self._token_budget - self._generated

    def _affordable(self, text: str, cursor: int) -> bool:
        # No budget declared means affordability is not in play and the
        # constant typo budget rules alone -- returning True here instead
        # would silently disable the excursion guard for every caller that
        # does not pass one.
        if self._token_budget is None:
            return False
        return (self._finish_tokens(text, cursor) + self.MARGIN_TOKENS
                <= self._remaining())

    def _excursion_exhausted(self) -> bool:
        """Depth alone no longer forces resolution -- affordability can
        keep a long, deliberate revision open past the typo budget."""
        return (self._depth() >= self.budget
                and not self._affordable(self.buffer.text, self.buffer.cursor))

    def _at_floor(self) -> bool:
        """True once only the compelled resolution route still fits."""
        if self._token_budget is None:
            return False
        return not self._affordable(self.buffer.text, self.buffer.cursor)

    def _excursions_open(self) -> bool:
        """Whether an off-path key is permitted at this position.

        Factored out because _forced_cur has to know it too: a caret move is
        only the LAST legal resort when no key is available, and pinning it
        when keys are open would forbid ordinary typing.
        """
        if self._at_floor():
            return False
        text, cursor = self.buffer.text, self.buffer.cursor
        affordable = self._affordable(
            text[:cursor] + "\0" + text[cursor:], cursor + 1
        )
        return ((self._depth() < self.budget or affordable)
                and not self._resolving
                and self._cooldown == 0
                and not self._stale_forced())

    def _stale_forced(self) -> bool:
        """Off-path for longer than a writer would leave it unnoticed.

        Edit-distance depth does not grow as you type past a divergence, so
        it cannot express "this has been wrong for a while". Without this the
        error is only forced at the very end, and the repair is a backspace
        run the length of everything typed since -- the observed
        types-the-line-then-deletes-it-all behaviour.

        The allowance scales with depth, because depth is what separates the
        two kinds of off-path stretch: typing past a typo leaves depth ~1
        while staleness grows, and must be forced promptly; a deliberate
        revision excursion grows depth with roughly every event, and a flat
        window the size of a typo guillotines it at event eleven -- Fix 1c's
        excursions could never actually start. The slack factor tolerates a
        draft-like excursion whose text half-matches the target (depth
        growing at half the event rate); anything staler than that is a
        divergence being typed past, not a revision. It cannot be 3: the
        typo case (depth 1) must still force within staleness_window + 3.
        """
        allowance = self.staleness_window + self.STALE_DEPTH_SLACK * self._depth()
        return self._depth() > 0 and self._stale > allowance

    def _forced_cur(self) -> int | None:
        """The pinned caret position when only a cursor move is sound.

        Returns the position <CUR:> must carry, or None. Three sources:
        resolving with the caret at 0 (BKSP would no-op forever), cooldown
        repayment with the caret mid-buffer (the needed key only extends the
        end), and the budget floor, where the caret must go to the divergence
        point so the tail can be removed in one SELDEL instead of N
        backspaces.
        """
        if self._ops is None:
            return None
        cursor, n = self.buffer.cursor, len(self.buffer.text)
        depth = self._depth()
        if self._resolving and depth > 0 and cursor == 0:
            return n
        if self._cooldown > 0 and depth == 0 and cursor < n:
            return n
        p = _common_prefix_len(self.buffer.text, self.target)
        if self._at_floor():
            if n > p and cursor != p:
                return p
            if n == p and cursor != n:
                return n
        # NOT DONE: pinning the caret to just past the divergence so one
        # backspace removes exactly the wrong character. Tried and reverted --
        # with the divergence at index 0 it pins to 1, and BKSP there deletes
        # the FRONT of the buffer, shifting everything so the divergence stays
        # at 0. Measured on motor-phase2: 281 cursor ops, 1015 events, on-path
        # 0/140, stalled -- delete-from-the-back turned into
        # delete-from-the-front. Routing repair to the cheap edit needs to know
        # the KIND of divergence (a wrong character to delete vs a missing one
        # to insert); the common-prefix index alone cannot tell them apart.
        # Caret at 0 with text and nothing else legal: BKSP no-ops there and no
        # key is offered off the divergence, so a caret move is the only
        # progress -- and it must be PINNED, or a sampler can hop 0 -> 0 for
        # ever. Pinning it to the divergence is where the needed key becomes
        # legal.
        if (self.buffer.text and cursor == 0 and p != 0
                and not self._excursions_open()):
            return p
        return None

    def _feasible_kinds(self, forced: int | None) -> set[str]:
        # SELDEL needs a nonempty range to delete; CUR:0 is always valid.
        if forced is not None:
            return {"cur"}
        kinds = {"cur"}
        if self.buffer.text:
            kinds.add("seldel")
        return kinds

    def _op_bounds(self, op: dict) -> tuple[int, int]:
        n = len(self.buffer.text)  # static for the whole op
        if op["kind"] == "cur":
            f = op["forced"]
            return (f, f) if f is not None else (0, n)
        if op["phase"] == "a":
            return 0, n - 1  # a < b <= n needs a <= n-1
        return op["a"] + 1, n

    def _start_op(self, node: dict, forced: int | None) -> None:
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
            # A caret move edits nothing, so it cannot pay down staleness and
            # cannot be allowed to repeat forever; only text changes reset it.
            self._hops += 1
        else:
            self.buffer._seldel(op["a"], int(op["digits"]))
            if self._depth() == 0:
                self._resolving = False
                self._cooldown = self.resolve_progress
            elif self._excursion_exhausted():
                self._resolving = True
        self._op = None
        self._slot = "gap"
        self._tick()

    def _tick(self) -> None:
        """One emitted event: staleness and the hop cap both reckon from
        PROGRESS, not from activity.

        Resetting the hop cap on any edit let a caret/backspace pair reset it
        every other event, so CUR-to-1 + BKSP could delete the buffer from the
        front for ever -- measured: 281 cursor ops, 1015 events, on-path 0/140,
        stalled. Neither of those events grows the correct prefix, so neither
        may clear the guard.
        """
        self._stale = 0 if self._depth() == 0 else self._stale + 1
        grown = _common_prefix_len(self.buffer.text, self.target)
        if grown > self._prefix:
            self._prefix = grown
            self._hops = 0

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
                elif self._excursion_exhausted():
                    self._resolving = True
                self._slot = "gap"
                self._tick()
            elif kind == "bksp":
                deleted = self.buffer.cursor > 0
                self.buffer._backspace()
                if deleted and self._depth() == 0:
                    self._resolving = False
                    self._cooldown = self.resolve_progress
                self._slot = "gap"
                self._tick()
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
        self._generated = generated
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

            if self._forced_cur() is not None:
                allowed = [self._to("op_cur", self._op_root_cur, device)]
            else:
                allowed = []
                # The needed key is whatever extends the correct prefix, and
                # that is typeable wherever the caret sits ON the divergence --
                # not only at the end of the buffer. Requiring cursor == n made
                # mid-buffer repair impossible to COMPLETE, which is why
                # deleting back to the mistake was the only route that worked;
                # on-path-at-the-end is just the special case where the
                # divergence IS the end.
                div = _common_prefix_len(text, self.target)
                if cursor == div and div < len(self.target):
                    needed = self.target[div]
                    allowed.append(
                        self._to(f"c:{needed}", self._per_char[needed], device)
                    )
                floor = self._at_floor()
                # An excursion is permitted either because it is a
                # character-level slip inside the small typo budget, or
                # because the remaining budget can still pay to undo it and
                # finish -- the affordability rule. The second is what makes a
                # semantic revision (10-50 characters of off-target wording)
                # representable at all; a constant of 4 forbids it outright.
                excursions_open = self._excursions_open()
                if excursions_open:
                    allowed.append(self._to("keys", self._all_keys, device))
                    if self._ops is not None:
                        roots = (self._op_root_all if text
                                 else self._op_root_cur)
                        allowed.append(
                            self._to(f"op_root:{bool(text)}", roots, device)
                        )
                elif (self._ops is not None and text and depth > 0
                      and self._hops < self.caret_hop_cap):
                    # Repair route, off-path. Reaching the divergence point
                    # with the caret is what lets one SELDEL undo a long
                    # excursion; without it the only repair is a backspace run
                    # as long as everything typed since the mistake -- the
                    # types-the-line-then-deletes-it-all behaviour.
                    allowed.append(
                        self._to("op_root_all", self._op_root_all, device)
                    )
                # A caret move edits no text, so it cannot break convergence
                # and must not share the typo gate: on-path it is the start of
                # every mid-buffer revision, and gating it there is what made
                # semantic revision unrepresentable. SELDEL stays gated -- it
                # deletes text, so it belongs with excursions. The hop cap is
                # what keeps a caret-only sampler terminating.
                if (self._ops is not None and depth == 0 and text
                        and not floor and not excursions_open
                        and self._hops < self.caret_hop_cap):
                    allowed.append(
                        self._to("op_cur", self._op_root_cur, device)
                    )
                if text and cursor > 0 and not (self._cooldown > 0 and depth == 0):
                    # No BKSP while repaying progress on-path: deleting
                    # correct text there would reopen the oscillation loop.
                    # At the floor with ops available, a BKSP is offered only
                    # if the state it creates can still pay for compelled
                    # resolution (the press costs 2 tokens): a genuine
                    # slip-fix passes, deleting correct prefix never does --
                    # without this, once the caret lands ON the divergence
                    # _forced_cur stops pinning and an adversarial sampler
                    # could backspace correct text until max_new_tokens.
                    bksp_ok = True
                    if floor and self._ops is not None:
                        after = text[: cursor - 1] + text[cursor:]
                        bksp_ok = (self._finish_tokens(after, cursor - 1)
                                   + self.MARGIN_TOKENS + 2
                                   <= self._remaining())
                    if bksp_ok:
                        allowed.append(self._to("bksp", self._bksp, device))
                if not allowed:
                    # Corner states must still offer something legal -- and it
                    # must make progress toward the target, never an off-path
                    # extension. Keyed on the divergence rather than on n: with
                    # mid-buffer carets now reachable, target[n] at a caret
                    # that is not the end would insert an unrelated character.
                    if cursor == div and div < len(self.target):
                        needed = self.target[div]
                        allowed.append(
                            self._to(f"c:{needed}", self._per_char[needed], device)
                        )
                    elif text and cursor > 0:
                        allowed.append(self._to("bksp", self._bksp, device))
                    elif self._ops is not None and text:
                        # Caret pinned at 0 with text present: BKSP no-ops, so
                        # the only progress is moving it.
                        allowed.append(
                            self._to("op_cur", self._op_root_cur, device)
                        )
                    else:
                        allowed.append(self._to("bksp", self._bksp, device))

        for ids in allowed:
            mask[:, ids] = 0
        return scores + mask
