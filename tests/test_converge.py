"""The convergence processor must make wrong final text unrepresentable.

Its contract: any generation that terminates (emits EOS) has typed the
target exactly; typos and corrections remain free within the excursion
budget; timing tokens are never restricted.
"""

import pytest

pytest.importorskip("torch")
import torch

from typeshi.converge import ConvergenceProcessor
from typeshi.serialize import special_tokens


class FakeTok:
    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(
            t for t in special_tokens() if t.endswith(">")
        )}
        self.eos_token_id = len(self.vocab)
        self.unk_token_id = len(self.vocab) + 1
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)


PROMPT_LEN = 7


def _mask_tokens(proc, tok, generated_ids):
    """Token strings legal at the state reached after `generated_ids`."""
    ids = torch.zeros((1, PROMPT_LEN + len(generated_ids)), dtype=torch.long)
    for j, g in enumerate(generated_ids):
        ids[0, PROMPT_LEN + j] = g
    scores = torch.zeros((1, tok.unk_token_id + 1))
    out = proc(ids, scores)
    legal = (out[0] > float("-inf")).nonzero().flatten().tolist()
    return {tok.id_to_token.get(i, "<EOS>") for i in legal}


def _tid(tok, t):
    return tok.vocab[t]


def test_on_path_offers_the_needed_key_and_excursions():
    tok = FakeTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    legal = _mask_tokens(proc, tok, [])
    assert any(t.startswith("<h:") for t in legal)      # the needed key
    assert any(t.startswith("<x:") for t in legal)      # typo excursions open
    assert not any(t.startswith("<DT:") for t in legal)
    assert "<EOS>" not in legal


def test_budget_exhaustion_forces_backspace_resolution():
    tok = FakeTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi", excursion_budget=2)
    # type two wrong chars (with gaps): buffer "xx", depth 2 == budget
    walk = [_tid(tok, "<x:5>"), _tid(tok, "<DT:5>"),
            _tid(tok, "<x:5>"), _tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert all(t.startswith("<BKSP:") for t in legal), legal


def test_resolution_cooldown_requires_on_path_progress():
    """After a forced resolution, excursions stay closed until progress is
    typed -- the guard against the type-delete-type oscillation the first
    live probe measured (1/5 converged at ~50% BKSP)."""
    tok = FakeTok()
    proc = ConvergenceProcessor(
        tok, PROMPT_LEN, "hi", excursion_budget=1, resolve_progress=1
    )
    walk = [
        _tid(tok, "<x:5>"), _tid(tok, "<DT:5>"),      # excursion to budget
        _tid(tok, "<BKSP:5>"), _tid(tok, "<DT:5>"),   # forced resolution
    ]
    legal = _mask_tokens(proc, tok, walk)
    # cooldown: ONLY the needed key -- no excursions, no BKSP churn
    assert legal and all(t.startswith("<h:") for t in legal), legal

    walk += [_tid(tok, "<h:5>"), _tid(tok, "<DT:5>")]  # progress repaid
    proc2 = ConvergenceProcessor(
        tok, PROMPT_LEN, "hi", excursion_budget=1, resolve_progress=1
    )
    legal = _mask_tokens(proc2, tok, walk)
    assert any(t.startswith("<x:") for t in legal)     # excursions reopen


def test_adversarial_sampler_still_converges():
    """The cooldown makes progress structural: even a sampler that always
    prefers WRONG keys terminates on the exact target within a linear
    token budget."""
    import random

    from typeshi.buffer import replay
    from typeshi.serialize import deserialize

    tok = FakeTok()
    target = "hi ho"
    rng = random.Random(0)
    proc = ConvergenceProcessor(
        tok, PROMPT_LEN, target, excursion_budget=2, resolve_progress=2
    )
    walk: list[int] = []
    terminated = False
    for _ in range(600):
        ids = torch.zeros((1, PROMPT_LEN + len(walk)), dtype=torch.long)
        for j, g in enumerate(walk):
            ids[0, PROMPT_LEN + j] = g
        scores = torch.zeros((1, tok.unk_token_id + 1))
        out = proc(ids, scores)[0]
        legal = (out > float("-inf")).nonzero().flatten().tolist()
        if tok.eos_token_id in legal:
            terminated = True
            break
        names = {i: tok.id_to_token.get(i, "") for i in legal}
        wrong = [i for i, n in names.items() if n.startswith(("<q:", "<z:"))]
        dts = [i for i, n in names.items() if n.startswith("<DT:")]
        pick = rng.choice(dts) if dts else (rng.choice(wrong) if wrong else rng.choice(legal))
        walk.append(pick)
    assert terminated, "adversarial walk failed to converge"
    text = "".join(tok.id_to_token[i] for i in walk)
    assert replay(deserialize(text)) == target


def test_eos_is_legal_exactly_when_buffer_equals_target():
    tok = FakeTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    # "h" typed -> gap position, not converged: no EOS
    legal = _mask_tokens(proc, tok, [_tid(tok, "<h:5>")])
    assert "<EOS>" not in legal and any(t.startswith("<DT:") for t in legal)

    proc2 = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    walk = [_tid(tok, "<h:5>"), _tid(tok, "<DT:5>"), _tid(tok, "<i:5>")]
    legal = _mask_tokens(proc2, tok, walk)
    assert "<EOS>" in legal


def test_timing_tokens_are_never_restricted_in_gap_position():
    tok = FakeTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    legal = _mask_tokens(proc, tok, [_tid(tok, "<x:5>")])  # mid-excursion gap
    dt_count = sum(1 for t in legal if t.startswith("<DT:"))
    from typeshi import config
    assert dt_count == config.TIME_BINS


def test_unsupported_target_char_raises_upfront():
    tok = FakeTok()
    with pytest.raises(ValueError, match="unsupported"):
        ConvergenceProcessor(tok, PROMPT_LEN, "héllo")


def test_simulated_walks_always_terminate_on_the_exact_target():
    """A noisy sampler under the mask: every terminated walk typed the
    target byte-for-byte, typos and all corrections included."""
    import random

    from typeshi.buffer import replay
    from typeshi.serialize import _encode_char, deserialize

    tok = FakeTok()
    target = "the cat."
    for trial in range(5):
        rng = random.Random(trial)
        proc = ConvergenceProcessor(tok, PROMPT_LEN, target, excursion_budget=3)
        walk: list[int] = []
        buf = ""  # test-side replica; keys append, BKSP truncates
        terminated = False
        for _ in range(400):
            ids = torch.zeros((1, PROMPT_LEN + len(walk)), dtype=torch.long)
            for j, g in enumerate(walk):
                ids[0, PROMPT_LEN + j] = g
            scores = torch.zeros((1, tok.unk_token_id + 1))
            out = proc(ids, scores)[0]
            legal = (out > float("-inf")).nonzero().flatten().tolist()
            if tok.eos_token_id in legal:  # a trained model stops when done
                terminated = True
                break
            # Model-like behaviour: mostly type what the text needs next
            # (or fix mistakes), with a 20% typo/noise rate. The mask's
            # guarantee is only that TERMINATED walks match; a sampler
            # with no intent to type is bounded by max_new_tokens instead.
            names = {i: tok.id_to_token.get(i, "") for i in legal}
            if buf == target[: len(buf)] and len(buf) < len(target):
                wanted = f"<{_encode_char(target[len(buf)])}:"
            else:
                wanted = "<BKSP:"
            preferred = [i for i, n in names.items() if n.startswith(wanted)]
            dts = [i for i, n in names.items() if n.startswith("<DT:")]
            if dts:
                pick = rng.choice(dts)
            elif preferred and rng.random() < 0.8:
                pick = rng.choice(preferred)
            else:
                pick = rng.choice(legal)
            walk.append(pick)
            name = names[pick]
            if name.startswith("<BKSP:"):
                buf = buf[:-1]
            elif not name.startswith("<DT:"):
                from typeshi.serialize import _decode_char

                buf += _decode_char(name[1:].rsplit(":", 1)[0])
        assert terminated, f"trial {trial} never converged in 400 tokens"
        text = "".join(tok.id_to_token[i] for i in walk)
        assert replay(deserialize(text)) == target


# ---------------------------------------------------------------------------
# Stage 2: <CUR:pos> / <SELDEL:a-b> under the guarantee. These ops are plain
# text through the base tokenizer, so the processor's digit-level machine is
# what makes an out-of-bounds position UNREPRESENTABLE, not merely unlikely.


class OpTok(FakeTok):
    """FakeTok plus the per-char pieces cursor ops decode through (mirrors
    tiny_tokenizer, where every text char is its own piece). The plain
    FakeTok has no encode(), which is exactly the stage-1-only contract."""

    def __init__(self):
        super().__init__()
        extra = sorted(set("<CUR:SELDEL->0123456789"))
        base = self.unk_token_id + 1
        for k, ch in enumerate(extra):
            self.vocab[ch] = base + k
        self.eos_token_id = base + len(extra)
        self.unk_token_id = base + len(extra) + 1
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    def encode(self, s, add_special_tokens=False):
        return [self.vocab[c] for c in s]


def test_stage1_tokenizer_without_atoms_never_offers_cursor_ops():
    tok = FakeTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    assert proc._ops is None  # no encode() -> ops stay masked out entirely


def test_cur_digits_are_masked_to_live_buffer_positions():
    """A cursor to 10 in a 1-char buffer must be unrepresentable: after
    '1' the only continuation is '>', because 10..19 all exceed len == 1."""
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    walk = [_tid(tok, "<h:5>"), _tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert "<" in legal                       # ops open with excursions
    walk += [tok.vocab[c] for c in "<CUR:"]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"0", "1"}                # 0 <= pos <= len(buffer) == 1
    walk.append(tok.vocab["1"])
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {">"}


def test_cur_completion_moves_the_caret_and_inserts_follow_it():
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    walk = [_tid(tok, "<h:5>"), _tid(tok, "<DT:5>")]
    walk += [tok.vocab[c] for c in "<CUR:0"]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {">"}                     # canonical: no leading zeros
    walk.append(tok.vocab[">"])
    legal = _mask_tokens(proc, tok, walk)     # a completed op is one event
    assert any(t.startswith("<DT:") for t in legal) and "<EOS>" not in legal
    assert proc.buffer.text == "h" and proc.buffer.cursor == 0
    walk += [_tid(tok, "<DT:5>"), _tid(tok, "<x:5>")]
    _mask_tokens(proc, tok, walk)
    assert proc.buffer.text == "xh"           # inserted AT the cursor


def test_seldel_needs_a_nonempty_buffer():
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi")
    walk = [tok.vocab["<"]]                   # empty buffer, op opened
    legal = _mask_tokens(proc, tok, walk)
    assert "C" in legal and "S" not in legal  # CUR:0 valid; no range to delete


def test_seldel_bounds_and_on_path_landing():
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi", excursion_budget=3)
    walk = []
    for c in "hix":                           # h, i on path; x an excursion
        walk += [_tid(tok, f"<{c}:5>"), _tid(tok, "<DT:5>")]
    walk += [tok.vocab[c] for c in "<SELDEL:"]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"0", "1", "2"}           # a < b <= 3 needs a <= 2
    walk.append(tok.vocab["2"])
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"-"}                     # 20.. all exceed 2; a=2 valid
    walk.append(tok.vocab["-"])
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"3"}                     # b in (a, len] == (2, 3]
    walk += [tok.vocab["3"]]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {">"}
    walk.append(tok.vocab[">"])
    legal = _mask_tokens(proc, tok, walk)
    assert proc.buffer.text == "hi" and proc.buffer.cursor == 2
    assert "<EOS>" in legal                   # landed exactly on the target


def test_seldel_to_budget_forces_cursor_to_end_resolution():
    """SELDEL can strand the caret at 0 while off-path, where BKSP no-ops;
    the mask must force <CUR:len> (and nothing else) so resolution can
    proceed, then BKSP-only until the buffer is back on-path."""
    tok = OpTok()
    proc = ConvergenceProcessor(
        tok, PROMPT_LEN, "abcd", excursion_budget=1, resolve_progress=1
    )
    walk = []
    for c in "abc":
        walk += [_tid(tok, f"<{c}:5>"), _tid(tok, "<DT:5>")]
    walk += [tok.vocab[c] for c in "<SELDEL:0-2>"]
    _mask_tokens(proc, tok, walk)
    assert proc.buffer.text == "c" and proc.buffer.cursor == 0
    walk += [_tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"<"}                     # forced: only the CUR opener
    walk += [tok.vocab[c] for c in "<CUR:"]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"1"}                     # exactly len(buffer)
    walk += [tok.vocab["1"], tok.vocab[">"], _tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert legal and all(t.startswith("<BKSP:") for t in legal)
    walk += [_tid(tok, "<BKSP:5>"), _tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)     # resolved -> cooldown: needed key
    assert legal and all(t.startswith("<a:") for t in legal)


def test_cooldown_with_mid_buffer_cursor_forces_return_to_end():
    """A mid-buffer SELDEL that lands on-path arms the cooldown with the
    caret inside the text, where the needed key would corrupt the buffer;
    repayment must route through cursor-to-end first."""
    tok = OpTok()
    proc = ConvergenceProcessor(
        tok, PROMPT_LEN, "abcd", excursion_budget=2, resolve_progress=1
    )
    walk = []
    for c in "abxc":                          # 'x' is a mid-word excursion
        walk += [_tid(tok, f"<{c}:5>"), _tid(tok, "<DT:5>")]
    walk += [tok.vocab[c] for c in "<SELDEL:2-3>"]
    _mask_tokens(proc, tok, walk)
    assert proc.buffer.text == "abc" and proc.buffer.cursor == 2
    walk += [_tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"<"}
    walk += [tok.vocab[c] for c in "<CUR:"]
    legal = _mask_tokens(proc, tok, walk)
    assert legal == {"3"}
    walk += [tok.vocab["3"], tok.vocab[">"], _tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert legal and all(t.startswith("<d:") for t in legal)
    walk += [_tid(tok, "<d:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert "<EOS>" in legal                   # buffer == target


def test_stage2_random_walks_converge_and_replay_to_target():
    """The guarantee with ops admitted: every terminated walk -- cursor
    jumps, range deletes, typos and all -- replays to the exact target."""
    import random

    from typeshi.buffer import replay
    from typeshi.serialize import _encode_char, deserialize

    tok = OpTok()
    target = "hi ho"
    for trial in range(3):
        rng = random.Random(trial)
        proc = ConvergenceProcessor(tok, PROMPT_LEN, target, excursion_budget=3)
        walk: list[int] = []
        terminated = False
        for _ in range(800):
            ids = torch.zeros((1, PROMPT_LEN + len(walk)), dtype=torch.long)
            for j, g in enumerate(walk):
                ids[0, PROMPT_LEN + j] = g
            scores = torch.zeros((1, tok.unk_token_id + 1))
            out = proc(ids, scores)[0]
            legal = (out > float("-inf")).nonzero().flatten().tolist()
            if tok.eos_token_id in legal:
                terminated = True
                break
            names = {i: tok.id_to_token.get(i, "") for i in legal}
            buf = proc.buffer.text
            if buf == target[: len(buf)] and len(buf) < len(target):
                wanted = f"<{_encode_char(target[len(buf)])}:"
            else:
                wanted = "<BKSP:"
            preferred = [i for i, n in names.items() if n.startswith(wanted)]
            dts = [i for i, n in names.items() if n.startswith("<DT:")]
            if dts:
                pick = rng.choice(dts)
            elif preferred and rng.random() < 0.85:
                pick = rng.choice(preferred)
            else:
                pick = rng.choice(legal)      # includes op openers and digits
            walk.append(pick)
        assert terminated, f"trial {trial} never converged in 800 tokens"
        text = "".join(tok.id_to_token[i] for i in walk)
        assert replay(deserialize(text)) == target


def test_continuation_window_seeds_buffer_and_opens_in_gap_slot():
    """Windowed generation: the processor resumes mid-essay. The first slot
    is GAP (the model emits the window-boundary <DT:> itself, as trained)
    and the needed key is the one after the written prefix."""
    tok = FakeTok()
    proc = ConvergenceProcessor(
        tok, PROMPT_LEN, "hi ho", written_so_far="hi ", cursor=3
    )
    legal = _mask_tokens(proc, tok, [])
    assert any(t.startswith("<DT:") for t in legal)   # boundary gap
    assert "<EOS>" not in legal                       # not converged yet

    walk = [_tid(tok, "<DT:5>")]                      # boundary gap emitted
    proc2 = ConvergenceProcessor(
        tok, PROMPT_LEN, "hi ho", written_so_far="hi ", cursor=3
    )
    legal = _mask_tokens(proc2, tok, walk)
    assert any(t.startswith("<h:") for t in legal)    # target[3] == 'h'


def test_continuation_eos_still_requires_full_target():
    tok = FakeTok()
    proc = ConvergenceProcessor(
        tok, PROMPT_LEN, "hi", written_so_far="h", cursor=1
    )
    walk = [_tid(tok, "<DT:5>"), _tid(tok, "<i:5>")]  # completes the target
    legal = _mask_tokens(proc, tok, walk)
    assert "<EOS>" in legal


def test_window_parse_trims_dangling_structure():
    from typeshi.generate import _parse_window

    good = "<h:5><DT:5><i:5>"
    assert len(_parse_window(good)) == 2
    assert len(_parse_window(good + "<DT:9>")) == 2          # dangling gap
    assert len(_parse_window(good + "<DT:9><CUR:1")) == 2    # mid-op cut
    assert _parse_window("<CUR:") == []


def test_window_shift_preserves_holds_and_gaps():
    from typeshi.events import Event
    from typeshi.generate import _shift_events

    ev = [Event.key("a", 10, 60), Event.key("b", 100, 130)]
    shifted = _shift_events(ev, 1000)
    assert [e.press_time for e in shifted] == [1010, 1100]
    assert shifted[0].release_time - shifted[0].press_time == 50
    assert shifted[1].press_time - shifted[0].press_time == 90
