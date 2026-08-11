import pytest

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from typeshi.serialize import deserialize

needs_network = pytest.mark.network

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


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


# ---- Offline mask tests (tiny tokenizer needs no network) ----------------


@pytest.fixture(scope="module")
def tiny_tok():
    from typeshi.tiny_tokenizer import build_tiny_tokenizer

    return build_tiny_tokenizer()


def _masked_scores(tiny_tok, n_generated: int):
    import torch

    from typeshi.constrain import TranscriptionGrammarProcessor

    prompt_len = 5
    proc = TranscriptionGrammarProcessor(tiny_tok, prompt_len)
    input_ids = torch.zeros((1, prompt_len + n_generated), dtype=torch.long)
    scores = torch.zeros((1, len(tiny_tok)))
    return proc(input_ids, scores)


def test_eos_illegal_at_stream_start(tiny_tok):
    out = _masked_scores(tiny_tok, 0)
    assert out[0, tiny_tok.eos_token_id].item() == float("-inf")


def test_eos_legal_in_event_slot_after_start(tiny_tok):
    out = _masked_scores(tiny_tok, 2)
    assert out[0, tiny_tok.eos_token_id].item() == 0.0


def test_eos_legal_in_gap_slot(tiny_tok):
    """TRL appends EOS right after the final event token -- a gap slot. The
    mask must allow what training taught, or a from-scratch model can never
    terminate. A stream ending event-then-EOS has no dangling <DT:>."""
    out = _masked_scores(tiny_tok, 1)
    assert out[0, tiny_tok.eos_token_id].item() == 0.0


def test_gap_slot_still_rejects_events_and_event_slot_rejects_dt(tiny_tok):
    event_id = tiny_tok.convert_tokens_to_ids("<a:50>")
    dt_id = tiny_tok.convert_tokens_to_ids("<DT:50>")
    gap = _masked_scores(tiny_tok, 1)
    assert gap[0, event_id].item() == float("-inf")
    assert gap[0, dt_id].item() == 0.0
    event = _masked_scores(tiny_tok, 2)
    assert event[0, dt_id].item() == float("-inf")
    assert event[0, event_id].item() == 0.0


def test_plain_text_char_masked_in_every_slot(tiny_tok):
    char_id = tiny_tok.convert_tokens_to_ids("a")
    for n in (0, 1, 2):
        out = _masked_scores(tiny_tok, n)
        assert out[0, char_id].item() == float("-inf")
