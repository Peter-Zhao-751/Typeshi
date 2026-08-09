import pytest

transformers = pytest.importorskip("transformers")

from typeshi.serialize import special_tokens
from typeshi.train_motor import build_peft_config, prepare_tokenizer

TINY = "hf-internal-testing/tiny-random-gpt2"

# Only the tokenizer tests need to download a model; the rest are pure logic.
needs_network = pytest.mark.network


@needs_network
def test_special_tokens_are_added_to_the_vocabulary():
    tok = prepare_tokenizer(TINY)
    for t in ["<BKSP>", "<DT:0>", "<KEY:SPC>"]:
        assert len(tok.tokenize(t)) == 1, f"{t} was split into multiple tokens"


@needs_network
def test_event_tokens_survive_encode_decode():
    tok = prepare_tokenizer(TINY)
    s = "<DT:5><KEY:a><HOLD:3><BKSP>"
    assert tok.decode(tok.encode(s), skip_special_tokens=False).replace(" ", "") == s


@needs_network
def test_vocabulary_grew_by_the_expected_amount():
    base = transformers.AutoTokenizer.from_pretrained(TINY)
    tok = prepare_tokenizer(TINY)
    # 3 entries ("<CUR:", "<SELDEL:", "<BKSP>" is whole) are prefixes, not tokens
    assert len(tok) >= len(base) + len(special_tokens()) - 3


def test_peft_config_targets_attention_projections():
    cfg = build_peft_config()
    assert cfg.r >= 16
    assert "q_proj" in cfg.target_modules and "v_proj" in cfg.target_modules


# --- backend selection is pure logic, so it needs no model download ---

@pytest.mark.parametrize(
    "cuda,bf16,mps,expect_dtype,expect_bf16",
    [
        (True, True, False, "bfloat16", True),    # the intended GPU target
        (True, False, False, "float16", False),   # older CUDA card
        (False, False, True, "float32", False),   # Apple Silicon
        (False, False, False, "float32", False),  # plain CPU
    ],
)
def test_backend_selection(cuda, bf16, mps, expect_dtype, expect_bf16):
    from typeshi.train_motor import select_backend

    got = select_backend(has_cuda=cuda, has_bf16=bf16, has_mps=mps)
    assert got["dtype"] == expect_dtype
    assert got["bf16"] is expect_bf16


def test_bfloat16_is_never_paired_with_device_map_off_cuda():
    """bf16 weights plus device_map='auto' segfault on Apple Silicon
    (torch 2.13/MPS); either alone is fine. Never emit that combination
    unless CUDA is present."""
    from typeshi.train_motor import select_backend

    for cuda, bf16, mps in [(False, False, True), (False, True, True), (False, False, False)]:
        got = select_backend(has_cuda=cuda, has_bf16=bf16, has_mps=mps)
        assert not (got["dtype"] == "bfloat16" and got["device_map"] == "auto")
