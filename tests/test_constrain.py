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
