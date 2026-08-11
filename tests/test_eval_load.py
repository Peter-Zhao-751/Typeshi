"""Routing only -- no weights are downloaded. The PoC checkpoint is a plain
model directory; AutoPeftModelForCausalLM requires adapter_config.json and
would fail on it, so load_checkpoint_model picks the class by that file."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from typeshi.eval.load import load_checkpoint_model

BACKEND = {"dtype": "float32", "device_map": None, "bf16": False}


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
