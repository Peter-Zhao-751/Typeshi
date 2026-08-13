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
