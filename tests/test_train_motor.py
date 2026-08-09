import pytest

transformers = pytest.importorskip("transformers")

from typeshi.serialize import special_tokens
from typeshi.train_motor import build_peft_config, prepare_tokenizer

TINY = "hf-internal-testing/tiny-random-gpt2"

pytestmark = pytest.mark.network


def test_special_tokens_are_added_to_the_vocabulary():
    tok = prepare_tokenizer(TINY)
    for t in ["<BKSP>", "<DT:0>", "<KEY:SPC>"]:
        assert len(tok.tokenize(t)) == 1, f"{t} was split into multiple tokens"


def test_event_tokens_survive_encode_decode():
    tok = prepare_tokenizer(TINY)
    s = "<DT:5><KEY:a><HOLD:3><BKSP>"
    assert tok.decode(tok.encode(s), skip_special_tokens=False).replace(" ", "") == s


def test_vocabulary_grew_by_the_expected_amount():
    base = transformers.AutoTokenizer.from_pretrained(TINY)
    tok = prepare_tokenizer(TINY)
    # 3 entries ("<CUR:", "<SELDEL:", "<BKSP>" is whole) are prefixes, not tokens
    assert len(tok) >= len(base) + len(special_tokens()) - 3


def test_peft_config_targets_attention_projections():
    cfg = build_peft_config()
    assert cfg.r >= 16
    assert "q_proj" in cfg.target_modules and "v_proj" in cfg.target_modules
