import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tok():
    from typeshi.tiny_tokenizer import build_tiny_tokenizer

    return build_tiny_tokenizer()


def test_default_config_is_about_19m(tok):
    from typeshi.train_tiny import build_tiny_model

    model = build_tiny_model("default", tok)
    n = sum(p.numel() for p in model.parameters())
    assert 18.5e6 < n < 20e6, f"{n/1e6:.2f}M"


def test_smoke_config_is_about_8m(tok):
    from typeshi.train_tiny import build_tiny_model

    model = build_tiny_model("smoke", tok)
    n = sum(p.numel() for p in model.parameters())
    assert 7.5e6 < n < 8.8e6, f"{n/1e6:.2f}M"


def test_kv_heads_explicit_and_mha(tok):
    """Qwen2Config defaults num_key_value_heads to 32, which silently builds
    a 29M model that crashes at the first forward. Spec §2 pins plain MHA."""
    from typeshi.train_tiny import TINY_CONFIGS, build_tiny_model

    for name, knobs in TINY_CONFIGS.items():
        assert knobs["num_key_value_heads"] == knobs["num_attention_heads"]
        model = build_tiny_model(name, tok)
        assert model.config.num_key_value_heads == knobs["num_attention_heads"]


def test_generation_config_terminates(tok):
    from typeshi.train_tiny import build_tiny_model

    model = build_tiny_model("smoke", tok)
    assert model.generation_config.eos_token_id == tok.eos_token_id
    assert model.generation_config.pad_token_id == tok.pad_token_id
    assert model.config.tie_word_embeddings is True


def test_encodable_flags_oov_prompts(tok):
    from typeshi.train_tiny import encodable

    assert encodable(tok, "<MODE:T><TARGET>plain ascii<PROCESS>")
    assert not encodable(tok, "curly “quote” target")
