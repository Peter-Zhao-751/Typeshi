"""Routing only -- no weights are downloaded. The PoC checkpoint is a plain
model directory; AutoPeftModelForCausalLM requires adapter_config.json and
would fail on it, so load_checkpoint_model picks the class by that file."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from typeshi.eval.load import load_checkpoint_model

BACKEND = {"dtype": "float32", "device_map": None, "bf16": False}

needs_network = pytest.mark.network

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_routes_peft_when_adapter_config_present(tmp_path, monkeypatch):
    import peft

    (tmp_path / "adapter_config.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        peft.AutoPeftModelForCausalLM, "from_pretrained",
        staticmethod(lambda path, **kw: calls.append(("peft", path)) or "peft-model"),
    )
    assert load_checkpoint_model(tmp_path, BACKEND) == "peft-model"
    assert calls == [("peft", tmp_path)]


def test_routes_plain_when_no_adapter_config(tmp_path, monkeypatch):
    import transformers

    calls = []
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained",
        staticmethod(lambda path, **kw: calls.append(("plain", path)) or "plain-model"),
    )
    assert load_checkpoint_model(tmp_path, BACKEND) == "plain-model"
    assert calls == [("plain", tmp_path)]


def _tiny_checkpoint_dir(tmp_path):
    """A minimal tiny-PoC checkpoint: our tokenizer + a qwen2 model config.
    The config.json is what springs the trap: AutoTokenizer cannot resolve
    the saved TokenizersBackend class name and falls back to model_type
    'qwen2', whose byte-level space handling eats every literal space."""
    import json

    from typeshi.tiny_tokenizer import build_tiny_tokenizer

    build_tiny_tokenizer().save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"],
        "vocab_size": 12909,
    }))
    return tmp_path


def test_checkpoint_tokenizer_preserves_spaces(tmp_path):
    from typeshi.eval.load import load_checkpoint_tokenizer

    tok = load_checkpoint_tokenizer(_tiny_checkpoint_dir(tmp_path))
    prompt = "<TARGET>a b c<PROCESS>"
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids, skip_special_tokens=False) == prompt


def test_checkpoint_tokenizer_guard_is_loud(tmp_path, monkeypatch):
    """If the loaded tokenizer mangles the probe, the eval must die loudly,
    never degrade into wrong-text rejections (that cost a full pilot cycle)."""
    import typeshi.eval.load as load_mod
    from typeshi.eval.load import load_checkpoint_tokenizer

    class Mangling:
        eos_token_id = 0
        pad_token_id = 1

        def __call__(self, text, **kw):
            return {"input_ids": [0]}

        def decode(self, ids, **kw):
            return "mangled"

    d = _tiny_checkpoint_dir(tmp_path)
    monkeypatch.setattr(
        load_mod, "_load_raw_tokenizer", lambda checkpoint: Mangling()
    )
    with pytest.raises(SystemExit):
        load_checkpoint_tokenizer(d)


@needs_network
def test_probe_passes_against_a_real_peft_style_tokenizer():
    """The untested branch: a PEFT checkpoint's tokenizer is loaded via
    AutoTokenizer (_load_raw_tokenizer's other path), on the SAME extended
    shape prepare_tokenizer() builds for Phase 1 (base tokenizer + added
    grammar tokens). The guard must not spuriously SystemExit a legitimate
    real-tokenizer eval -- that would be the exact failure class this task
    exists to prevent, just flipped onto the branch nothing else covers."""
    from typeshi.eval.load import _probe_ok
    from typeshi.train_motor import prepare_tokenizer

    tok = prepare_tokenizer(TINY)
    assert _probe_ok(tok)
