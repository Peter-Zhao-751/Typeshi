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

Stage 1 is linear: <CUR:>/<SELDEL:> are masked out entirely, so all editing
is end-of-buffer and "off-path depth" is simply the suffix beyond the common
prefix. Stage 2 re-admits cursor ops behind a digit-level state machine with
positions validated against the live buffer; until then composition keeps
the transcription event set (typos + backspace corrections), which the
probes showed carries most of the behaviour (cursor ops are ~1% of real
composition events).

Termination: BKSP is reachable from every state and the next-needed key is
always sampleable once resolved, so buffer == target is always reachable;
max_new_tokens bounds the walk, and a generation that never converges is a
FAILED attempt for the caller to count, never a silently wrong text.
"""

from __future__ import annotations

from typeshi.buffer import TextBuffer
from typeshi.serialize import _encode_char, special_tokens


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class ConvergenceProcessor:
    """LogitsProcessor guaranteeing the stream types `target` exactly.

    Batch size 1, matching generate(). `excursion_budget` is the maximum
    off-path depth in characters before the mask offers only BKSP.
    """

    def __init__(self, tok, prompt_len: int, target: str,
                 excursion_budget: int = 4, resolve_progress: int = 2) -> None:
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
        # progress, so convergence is bounded, adversarial samplers included.
        self.resolve_progress = resolve_progress
        self._resolving = False
        self._cooldown = 0
        self.buffer = TextBuffer()
        self._consumed = 0  # generated tokens already applied to the buffer

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

    def _to(self, name: str, tensor, device):
        key = (name, device)
        if key not in self._dev:
            self._dev[key] = tensor.to(device)
        return self._dev[key]

    def _depth(self) -> int:
        text = self.buffer.text
        return len(text) - _common_prefix_len(text, self.target)

    def _apply_new_tokens(self, ids: list[int]) -> None:
        """Replays committed tokens and advances the guard state.

        State lives HERE, derived from the token stream, not in __call__ --
        the mask for a position must depend only on what was emitted before
        it, never on how many times the processor happened to be invoked.

        The cooldown arms on ANY backspace that lands on-path, not just
        budget-forced ones: shallow type-one-wrong/delete loops below the
        budget are otherwise free to oscillate forever, and so is deleting
        correct text and retyping it.
        """
        for i in ids:
            kind = self._id_kind.get(i)
            if kind == "key":
                self.buffer._insert(self._id_char[i])
                if self._depth() == 0 and self._cooldown > 0:
                    self._cooldown -= 1
                elif self._depth() >= self.budget:
                    self._resolving = True
            elif kind == "bksp":
                self.buffer._backspace()
                if self._depth() == 0:
                    self._resolving = False
                    self._cooldown = self.resolve_progress
            # dt / anything else: no buffer effect

    def __call__(self, input_ids, scores):
        import torch

        generated = input_ids.shape[1] - self.prompt_len
        new = input_ids[0, self.prompt_len + self._consumed:].tolist()
        self._apply_new_tokens(new)
        self._consumed = generated

        mask = torch.full_like(scores, float("-inf"))
        device = scores.device

        if generated % 2 == 1:
            # GAP position: timing is free; ending is earned.
            allowed = [self._to("dt", self._dt, device)]
            if self.buffer.text == self.target and self._eos.numel():
                allowed.append(self._to("eos", self._eos, device))
        else:
            # EVENT position. All guard state was advanced token-by-token in
            # _apply_new_tokens; this branch only reads it.
            text = self.buffer.text
            depth = self._depth()

            allowed = []
            if depth == 0 and len(text) < len(self.target):
                needed = self.target[len(text)]
                allowed.append(
                    self._to(f"c:{needed}", self._per_char[needed], device)
                )
            excursions_open = (
                depth < self.budget and not self._resolving and self._cooldown == 0
            )
            if excursions_open:
                allowed.append(self._to("keys", self._all_keys, device))
            if text and not (self._cooldown > 0 and depth == 0):
                # No BKSP while repaying progress on-path: deleting correct
                # text there would reopen the oscillation loop.
                allowed.append(self._to("bksp", self._bksp, device))
            if not allowed:
                # Corner states must still offer something legal: type the
                # next needed char if the target is unfinished, else undo one
                # char (retype cycle) -- never an off-path extension.
                if len(text) < len(self.target):
                    needed = self.target[len(text)]
                    allowed.append(
                        self._to(f"c:{needed}", self._per_char[needed], device)
                    )
                else:
                    allowed.append(self._to("bksp", self._bksp, device))

        for ids in allowed:
            mask[:, ids] = 0
        return scores + mask
