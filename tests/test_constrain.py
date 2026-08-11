import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from typeshi.serialize import deserialize, special_tokens

needs_network = pytest.mark.network

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class FakeTok:
    """Just enough tokenizer for TranscriptionGrammarProcessor: an id per
    registered special token, plus an EOS. Keeps the grammar tests offline --
    the inverted-EOS bug shipped precisely because the only tests of this
    module needed network and were always skipped."""

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(
            t for t in special_tokens() if t.endswith(">")
        )}
        self.eos_token_id = len(self.vocab)
        self.unk_token_id = len(self.vocab) + 1
        self.id_to_token = {i: t for t, i in self.vocab.items()}

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)


def _allowed(proc, tok, n_generated):
    """Token strings the mask permits at a position with n_generated tokens."""
    import torch

    vocab_size = tok.unk_token_id + 1
    ids = torch.zeros((1, 7 + n_generated), dtype=torch.long)  # prompt_len=7
    scores = torch.zeros((1, vocab_size))
    out = proc(ids, scores)
    legal = (out[0] > float("-inf")).nonzero().flatten().tolist()
    return {tok.id_to_token.get(i, "<EOS>") for i in legal}


def test_event_position_allows_keystrokes_never_eos_or_dt():
    from typeshi.constrain import TranscriptionGrammarProcessor

    tok = FakeTok()
    proc = TranscriptionGrammarProcessor(tok, prompt_len=7)
    for n in (0, 2, 8):  # even = event position
        allowed = _allowed(proc, tok, n)
        assert any(t.startswith("<a:") for t in allowed)
        assert any(t.startswith("<BKSP:") for t in allowed)
        assert not any(t.startswith("<DT:") for t in allowed)
        # EOS here would end the stream on a dangling <DT:>, which
        # deserialize rejects -- the exact inversion that shipped.
        assert "<EOS>" not in allowed


def test_gap_position_allows_dt_and_eos_only():
    from typeshi.constrain import TranscriptionGrammarProcessor

    tok = FakeTok()
    proc = TranscriptionGrammarProcessor(tok, prompt_len=7)
    for n in (1, 3, 9):  # odd = gap position
        allowed = _allowed(proc, tok, n)
        assert "<EOS>" in allowed  # the final keystroke has no gap
        assert any(t.startswith("<DT:") for t in allowed)
        assert not any(t.startswith("<a:") for t in allowed)
        assert not any(t.startswith("<BKSP:") for t in allowed)


def test_every_maskable_stop_point_parses():
    """Any stream built from mask-legal choices that ends at EOS must
    deserialize -- the mask and the parser must agree about where streams
    can end."""
    from typeshi.constrain import TranscriptionGrammarProcessor

    tok = FakeTok()
    proc = TranscriptionGrammarProcessor(tok, prompt_len=7)
    stream = []
    for n in range(7):  # walk: event, gap, event, gap...
        allowed = _allowed(proc, tok, n)
        if "<EOS>" in allowed:
            events = deserialize("".join(stream))  # stop here must be legal
            assert events, "empty stream should not offer EOS"
        pick = sorted(t for t in allowed if t != "<EOS>")[0]
        stream.append(pick)


@needs_network
def test_constrained_generation_is_always_grammar_legal():
    """Even a RANDOM model must emit a parseable alternating stream under the
    mask -- that is the whole point: grammar failures become impossible, so
    eval failures are about content and timing."""
    import torch
    from transformers import AutoModelForCausalLM, LogitsProcessorList

    from typeshi.constrain import TranscriptionGrammarProcessor
    from typeshi.train_motor import prepare_tokenizer

    tok = prepare_tokenizer(TINY)
    model = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    model.resize_token_embeddings(len(tok))

    prompt = "<MODE:T><WPM:12><ECOR:0><EUNC:0><REV:0><TARGET>hi<PROCESS>"
    inputs = tok(prompt, return_tensors="pt")
    torch.manual_seed(0)
    out = model.generate(
        **inputs, do_sample=True, max_new_tokens=40,
        pad_token_id=tok.pad_token_id,
        logits_processor=LogitsProcessorList(
            [TranscriptionGrammarProcessor(tok, inputs["input_ids"].shape[1])]
        ),
    )
    import re
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                      skip_special_tokens=False)
    if tok.eos_token and tok.eos_token in text:
        text = text.split(tok.eos_token)[0]
    text = re.sub(r"<DT:\d+>$", "", text)  # budget cutoff may land on a gap
    events = deserialize(text)   # must not raise
    assert len(events) >= 1


@needs_network
def test_constrained_mode_rejects_composition():
    from typeshi.generate import generate
    from typeshi.labels import SessionLabels

    with pytest.raises(ValueError):
        generate(None, None, "x", SessionLabels(60, 0, 0, 0),
                 mode="composition", constrained=True)
