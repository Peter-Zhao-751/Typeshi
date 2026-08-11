"""The tiny tokenizer must be byte-exact and closed-world: every grammar
token one ID, decode with no joiners, OOV raises. All properties are
checked on BOTH the in-memory tokenizer and its save/load round-trip --
added-token serialization is exactly where decode behavior drifts, and a
drift here would surface only after an overnight run (spec §3 req. 4)."""

import pytest

pytest.importorskip("transformers")

from typeshi.dataset import build_prompt
from typeshi.labels import SessionLabels
from typeshi.serialize import special_tokens

REGISTERED = [t for t in special_tokens() if t.endswith(">")]


@pytest.fixture(scope="module")
def fresh():
    from typeshi.tiny_tokenizer import build_tiny_tokenizer

    return build_tiny_tokenizer()


@pytest.fixture(scope="module")
def reloaded(fresh, tmp_path_factory):
    import json

    from typeshi.eval.load import load_checkpoint_tokenizer

    d = tmp_path_factory.mktemp("tiny-tok")
    fresh.save_pretrained(d)
    # A real checkpoint directory also has config.json (model_type: "qwen2"),
    # which hijacks bare AutoTokenizer.from_pretrained() into Qwen2Tokenizer
    # and silently eats every space (docs/results-tiny-poc.md §6.2) -- write
    # it here so this fixture reproduces the trap instead of sailing past it.
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "vocab_size": 12_909,
    }))
    # load_checkpoint_tokenizer, not bare AutoTokenizer: it's what every real
    # checkpoint consumer (eval, playground) uses, and it's the only path
    # that resolves the generic wrapper correctly instead of Qwen2Tokenizer.
    return load_checkpoint_tokenizer(d)


@pytest.fixture(scope="module", params=["fresh", "reloaded"])
def tok(request, fresh, reloaded):
    return {"fresh": fresh, "reloaded": reloaded}[request.param]


def test_vocab_size_is_exact(tok):
    assert len(tok) == 97 + 2 + 12_810 == 12_909


def test_every_grammar_token_is_single_id(tok):
    for t in REGISTERED:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"{t} split into {len(ids)} ids"


def test_byte_exact_roundtrip_of_real_prompt_and_completion(tok):
    labels = SessionLabels(72.0, 0.02, 0.01, 0.0)
    prompt = build_prompt("I was trying to remember.", labels, "transcription")
    completion = "<I:56><DT:54><SPC:53><DT:56><w:53><DT:59><a:49>"
    for text in (prompt, completion, prompt + completion):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert tok.decode(ids, skip_special_tokens=False) == text


def test_oov_char_raises(tok):
    with pytest.raises(Exception):
        tok("café")


def test_eos_and_pad_distinct_and_stable(fresh, reloaded):
    assert fresh.eos_token_id is not None
    assert fresh.pad_token_id is not None
    assert fresh.eos_token_id != fresh.pad_token_id
    assert fresh.eos_token_id == reloaded.eos_token_id
    assert fresh.pad_token_id == reloaded.pad_token_id
