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
    # Resolution stays compelled -- no free keys, no ending -- but it may now
    # route through an op: one SELDEL undoes a long excursion where BKSP-only
    # forces a deletion run as long as the excursion itself.
    assert any(t.startswith("<BKSP:") for t in legal)
    assert not any(t.startswith("<a:") or t.startswith("<z:") for t in legal)
    assert "<EOS>" not in legal
    walk += [_tid(tok, "<BKSP:5>"), _tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)     # resolved -> cooldown: needed key
    assert any(t.startswith("<a:") for t in legal)
    assert not any(t.startswith("<z:") for t in legal)


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
    # Cooldown repayment offers the needed key and no other key; a caret move
    # edits nothing and is hop-capped, so it no longer has to be excluded.
    assert any(t.startswith("<d:") for t in legal)
    assert not any(t.startswith("<z:") or t.startswith("<a:") for t in legal)
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


# --- Fix 1/1b/1c: on-path cursor ops, staleness, affordability ----------

def _type(tok, s):
    """Token walk that types `s`, one <c:5><DT:5> pair per character."""
    from typeshi.serialize import _encode_char

    walk = []
    for ch in s:
        walk += [_tid(tok, f"<{_encode_char(ch)}:5>"), _tid(tok, "<DT:5>")]
    return walk


def test_cursor_ops_are_offered_on_path_even_when_excursions_are_closed():
    """Fix 1: a CUR op moves the caret and edits no text, so it cannot break
    convergence and must not share the typo-excursion gate. With the gate
    shut, a caret move is currently unrepresentable while typing on-path."""
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi there",
                                excursion_budget=0)
    walk = _type(tok, "hi")
    legal = _mask_tokens(proc, tok, walk)
    assert not any(t.startswith("<x:") for t in legal), "excursions shut"
    assert "<" in legal, "on-path caret move must be reachable"


def test_caret_hops_without_text_progress_are_capped():
    """Fix 1, part 2: ungating CUR must not admit an endless caret walk.

    Hops to the END, so the needed key stays legal throughout: the cap may
    only close a caret move that is a luxury, never one that is the sole
    legal progress (caret stranded at 0, where BKSP no-ops, is pinned by
    _forced_cur instead and must stay reachable).
    """
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hi there",
                                excursion_budget=0, caret_hop_cap=2)
    walk = _type(tok, "hi")
    for _ in range(2):  # two no-op hops, no text change between them
        walk += [tok.vocab[c] for c in "<CUR:2>"] + [_tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert "<" not in legal, "caret hops must be capped without progress"
    assert any(t.startswith("<SPC:") for t in legal), \
        "capping the caret must not strand the run"


def test_stale_off_path_run_is_forced_into_repair():
    """Fix 1b: edit-distance depth does not grow as you type past a typo, so
    nothing forces a fix. Staleness -- events since the buffer was last a
    clean prefix -- is what must force it, well before the end."""
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hello world",
                                excursion_budget=50, staleness_window=4)
    walk = _type(tok, "helo")            # typo at index 3, depth stays 1
    walk += _type(tok, " world")          # keeps typing, never repairs
    legal = _mask_tokens(proc, tok, walk)
    assert not any(t.startswith("<z:") for t in legal), \
        "a stale off-path run must stop offering free keys"
    assert "<" in legal or any(t.startswith("<BKSP:") for t in legal), \
        "repair moves must remain available"


def test_long_excursion_is_allowed_while_the_budget_can_still_pay():
    """Fix 1c: a semantic revision means 10-50 chars of off-target text. A
    constant budget of 4 forbids it; affordability should permit it when the
    remaining budget can still undo it and finish."""
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hello world",
                                excursion_budget=4, token_budget=4000,
                                staleness_window=10_000)
    walk = _type(tok, "hello") + _type(tok, "XXXXXXXXXX")  # 10 off-path chars
    legal = _mask_tokens(proc, tok, walk)
    assert any(t.startswith("<z:") for t in legal), \
        "an affordable excursion must stay open past the typo budget"


def test_deliberate_excursion_outlives_the_staleness_window():
    """Fix 1b vs 1c: staleness closes free keys after ~staleness_window
    off-path events, but a semantic revision IS 10-50 off-path events, so at
    the shipped default of 10 the behaviour Fix 1c permits can never occur.

    The two cases staleness must separate are distinguishable by depth: typing
    past a typo leaves depth ~1 while staleness grows; a deliberate excursion
    grows depth with every character. Scaling the allowance with depth keeps
    the typo case forced (the depth-1 test above) without guillotining the
    revision at event eleven.
    """
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hello world",
                                excursion_budget=4, token_budget=4000)
    walk = _type(tok, "hello") + _type(tok, "X" * 30)  # default staleness=10
    legal = _mask_tokens(proc, tok, walk)
    assert any(t.startswith("<z:") for t in legal), \
        "an affordable deliberate excursion must survive the staleness window"


def test_excursion_is_refused_when_the_budget_cannot_pay_to_finish():
    """The other half: affordability must actually bind when it runs out.

    excursion_budget=0 so the small typo allowance cannot open the excursion
    on its own -- affordability is the only thing under test here.
    """
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hello world",
                                excursion_budget=0, token_budget=45,
                                staleness_window=10_000)
    walk = _type(tok, "hello")            # 10 tokens spent, 35 left
    legal = _mask_tokens(proc, tok, walk)
    assert not any(t.startswith("<z:") for t in legal), \
        "off-path keys must close once the budget cannot pay to undo them"
    assert any(t.startswith("<SPC:") for t in legal), "the needed key stays legal"


def test_at_the_budget_floor_the_cheap_resolution_is_compelled():
    """The guarantee rests on the CHEAPEST route the mask can COMPEL.

    Authorising an excursion because CUR+SELDEL could undo it in 2 events,
    then letting the model backspace 10 times instead, strands the run with
    no budget and a wrong buffer. At the floor the mask must force the cheap
    route rather than merely offer it.
    """
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hello world",
                                excursion_budget=50, token_budget=1000,
                                staleness_window=10_000)
    walk = _type(tok, "hello") + _type(tok, "XXXXXXXXXX")
    proc._token_budget = len(walk) + 34   # just enough for CUR+SELDEL+finish
    legal = _mask_tokens(proc, tok, walk)
    assert "<" in legal, "the cheap route must be offered at the floor"
    assert not any(t.startswith("<BKSP:") for t in legal), \
        "the expensive route must be closed once it is unaffordable"


def test_at_the_floor_bksp_cannot_eat_the_correct_prefix():
    """The compelled-route guarantee leaked AFTER the caret landed on the
    divergence: _forced_cur stops pinning there, and the ordinary branch
    offered BKSP alongside the ops, so an adversarial sampler at the floor
    could backspace correct prefix characters until max_new_tokens. A floor
    BKSP is legal only if the state it creates can still pay for compelled
    resolution -- deleting a correct character never can."""
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hello world",
                                excursion_budget=50, token_budget=10_000,
                                staleness_window=10_000)
    walk = _type(tok, "hello") + _type(tok, "XXXXXXXXXX")
    walk += [tok.vocab[c] for c in "<CUR:5>"] + [_tid(tok, "<DT:5>")]
    _mask_tokens(proc, tok, walk)  # replay to the caret-at-divergence state
    proc._token_budget = (proc._generated
                          + proc._finish_tokens(proc.buffer.text,
                                                proc.buffer.cursor)
                          + proc.MARGIN_TOKENS - 1)  # one short: the floor
    legal = _mask_tokens(proc, tok, walk)
    assert "<" in legal, "the cheap route must stay open"
    assert not any(t.startswith("<BKSP:") for t in legal), \
        "backspacing correct prefix at the floor breaks the budget claim"


def test_needed_key_is_offered_mid_buffer_at_the_divergence():
    """Repairing in place requires typing the right character where it is
    missing. The mask only offered the needed key with the caret at the very
    end, so a mid-buffer repair could never be completed -- which is why
    deleting back to the mistake was the only route that worked."""
    tok = OpTok()
    proc = ConvergenceProcessor(tok, PROMPT_LEN, "hello world",
                                excursion_budget=0)
    # Type "hllo" -- 'e' missing at index 1 -- then park the caret at the gap.
    walk = _type(tok, "hllo")
    walk += [tok.vocab[c] for c in "<CUR:1>"] + [_tid(tok, "<DT:5>")]
    legal = _mask_tokens(proc, tok, walk)
    assert proc.buffer.cursor == 1
    assert any(t.startswith("<e:") for t in legal), \
        "the character that extends the correct prefix must be typeable here"
