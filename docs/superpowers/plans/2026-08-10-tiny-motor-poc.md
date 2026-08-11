# Tiny Motor Model PoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A from-scratch ~19M-parameter causal LM that emits token-format-v2 typing
streams, trained on this Mac and scored by the existing five-gate eval — proving the
dataset, format, constrained decoder, and eval plumbing end to end.

**Architecture:** Stock `Qwen2ForCausalLM` with a tiny config + a custom 12,909-token
char-level tokenizer, trained with the same SFTTrainer prompt/completion recipe as
`train_motor.py`. Three small edits to existing code (eval loader fallback, eval OOV
skip, EOS-in-gap-slot legalization in `constrain.py`). Spec:
`docs/superpowers/specs/2026-08-10-tiny-motor-poc-design.md`.

**Tech Stack:** PyTorch (fp32 on MPS via `select_backend()`), transformers/`tokenizers`,
TRL SFTTrainer, HF datasets, pytest. Run everything with `uv run`.

## Global Constraints

- Vocab is exactly **12,909**: 97 text chars + `<EOS>` + `<PAD>` + 12,810 grammar tokens. **No `<UNK>` anywhere.**
- The WordLevel model is constructed **without** an `unk_token` — that is what makes OOV encoding raise (spec §3 req. 3).
- `num_key_value_heads` is always set explicitly (= `num_attention_heads`; Qwen2Config's default of 32 silently builds a broken model).
- Loss on completion only (TRL prompt/completion columns — never collapse to one text field).
- Train only `<MODE:T>` examples; copy `split.json` into every checkpoint.
- `--limit` subsets are **seeded random samples**, never head-of-file slices.
- fp32 on MPS via the existing `select_backend()`; never bf16+device_map on MPS.
- Every tokenizer property must hold on the `save_pretrained → AutoTokenizer.from_pretrained` round-trip copy, not just in memory.
- Tests stay offline by default; anything that trains for minutes is marked `slow` and gated by `TYPESHI_SLOW_TESTS=1` (mirroring the existing `network` marker).
- Long runs go under `caffeinate -dim` with `--save-steps` checkpointing.

---

### Task 1: Tiny char-level tokenizer

**Files:**
- Create: `src/typeshi/tiny_tokenizer.py`
- Test: `tests/test_tiny_tokenizer.py`

**Interfaces:**
- Consumes: `typeshi.serialize.special_tokens()` (existing).
- Produces: `TEXT_CHARS: list[str]` (97 chars), `EOS = "<EOS>"`, `PAD = "<PAD>"`,
  `build_tiny_tokenizer() -> PreTrainedTokenizerFast` (len 12,909). Tasks 2–5 use all of these.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tiny_tokenizer.py
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
    from transformers import AutoTokenizer

    d = tmp_path_factory.mktemp("tiny-tok")
    fresh.save_pretrained(d)
    return AutoTokenizer.from_pretrained(d)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tiny_tokenizer.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.tiny_tokenizer'`

- [ ] **Step 3: Implement the tokenizer**

```python
# src/typeshi/tiny_tokenizer.py
"""Char-level tokenizer for the tiny motor PoC (spec 2026-08-10, §3).

Grammar tokens are single vocabulary entries, mirroring prepare_tokenizer()
on Qwen; <TARGET> text encodes one char per token, which turns target-copying
into a monotonic char->key attention pattern. Built programmatically -- no
vocab files to maintain."""

from __future__ import annotations

from typeshi.serialize import special_tokens

# Printable ASCII 0x20-0x7E plus newline and tab: the 97 characters that can
# appear in prompt text. Everything else must RAISE at encode time (the build
# gates typed chars but never the target sentence -- spec §3 OOV containment).
TEXT_CHARS = [chr(c) for c in range(0x20, 0x7F)] + ["\n", "\t"]

EOS = "<EOS>"
PAD = "<PAD>"


def build_tiny_tokenizer():
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(TEXT_CHARS)}
    # No unk_token, deliberately: WordLevel then raises on out-of-vocab input
    # instead of silently mapping it. Wiring an unk would violate spec §3.3.
    inner = Tokenizer(models.WordLevel(vocab))
    # (?s) so '.' matches newline; every char becomes its own piece.
    inner.pre_tokenizer = pre_tokenizers.Split(Regex(r"(?s)."), behavior="isolated")
    # The default decode joins word-level tokens with spaces; deserialize()
    # rejects stray whitespace by design, so decode must fuse byte-exactly.
    inner.decoder = decoders.Fuse()

    tok = PreTrainedTokenizerFast(
        tokenizer_object=inner,
        eos_token=EOS,
        pad_token=PAD,
        clean_up_tokenization_spaces=False,
    )
    tok.add_special_tokens({"eos_token": EOS, "pad_token": PAD})
    whole = [t for t in special_tokens() if t.endswith(">")]
    tok.add_tokens(whole, special_tokens=True)
    return tok
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tiny_tokenizer.py -q`
Expected: PASS (all 10: 5 tests × fresh/reloaded). If `test_byte_exact_roundtrip`
fails with spaces between tokens, the `decoders.Fuse()` line is missing or the
wrapper is cleaning spaces — do not "fix" by stripping in the test.

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/tiny_tokenizer.py tests/test_tiny_tokenizer.py
git commit -m "feat: char-level tiny tokenizer with byte-exact decode"
```

---

### Task 2: EOS legal in the gap slot (`constrain.py`)

TRL appends EOS directly after the completion's final event token — a **gap**
position under the alternating grammar — so every model is trained to emit EOS
exactly where the current mask forbids it. A from-scratch model cannot terminate
without this fix; it also benefits Phase 1 (Qwen's appended `<|im_end|>` lands in
the same slot).

**Files:**
- Modify: `src/typeshi/constrain.py:58-71` (`TranscriptionGrammarProcessor.__call__` and class docstring)
- Test: `tests/test_constrain.py` (add offline tests; existing network tests untouched)

**Interfaces:**
- Consumes: `build_tiny_tokenizer()` from Task 1 (gives the tests an offline tokenizer).
- Produces: mask behavior relied on by Tasks 5, 8–10: EOS legal in *both* slots once ≥1 token was generated, illegal at position zero.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_constrain.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify the gap-slot one fails**

Run: `uv run pytest tests/test_constrain.py -q`
Expected: `test_eos_legal_in_gap_slot` FAILS (`-inf == 0.0`); the other new tests
PASS (they pin current behavior); network tests skip.

- [ ] **Step 3: Apply the fix**

In `src/typeshi/constrain.py`, replace the `else` branch of `__call__`:

```python
        else:
            # Gap position. EOS is legal here too: TRL appends EOS directly
            # after the completion's final event token, i.e. in THIS slot, so
            # the mask must allow what training taught -- a from-scratch model
            # has no pretrained prior to terminate from anywhere else. The
            # stream then ends event-then-EOS with no dangling <DT:>.
            allowed = self.dt_ids
            if self.eos_ids.numel():
                allowed = torch.cat([allowed, self.eos_ids])
```

And update the class docstring's last sentence to:

```python
    Alternates between event tokens (<c:h>, <BKSP:h>) and gap tokens (<DT:k>);
    EOS becomes legal in BOTH positions once at least one event was emitted --
    training places it in the gap slot (TRL appends it after the final event
    token), and either ending leaves no dangling <DT:>.
```

- [ ] **Step 4: Run the full test file**

Run: `uv run pytest tests/test_constrain.py -q`
Expected: all offline tests PASS, network tests skip.

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/constrain.py tests/test_constrain.py
git commit -m "fix: legalize EOS in the gap slot -- TRL trains it there"
```

---

### Task 3: `train_tiny.py`

**Files:**
- Create: `src/typeshi/train_tiny.py`
- Test: `tests/test_train_tiny.py`

**Interfaces:**
- Consumes: `build_tiny_tokenizer()` (Task 1), `typeshi.train_motor._detect_backend`, `typeshi.config`.
- Produces: `TINY_CONFIGS: dict[str, dict]` (keys `"default"`, `"smoke"`),
  `build_tiny_model(name: str, tok) -> Qwen2ForCausalLM`,
  `encodable(tok, text: str) -> bool`, and the CLI
  `python -m typeshi.train_tiny --data … --out … --config {default,smoke} --epochs F --batch N --accum N --lr F --limit N --save-steps N --resume --seed N`.
  Tasks 5 and 7–10 rely on the CLI; the checkpoint directory it writes loads via
  `AutoTokenizer` + `AutoModelForCausalLM`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_tiny.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_tiny.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.train_tiny'`

- [ ] **Step 3: Implement**

```python
# src/typeshi/train_tiny.py
"""Tiny motor PoC: from-scratch ~19M causal LM on the Phase-1 export.

Mirrors train_motor.py's recipe -- SFTTrainer with prompt/completion columns
(loss on completion only), <MODE:T> filter, split.json bound to the
checkpoint -- with a from-scratch Qwen2 config and the char-level tiny
tokenizer instead of a pretrained base + LoRA. Spec: docs/superpowers/specs/
2026-08-10-tiny-motor-poc-design.md."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from typeshi import config

# num_key_value_heads is explicit everywhere: Qwen2Config defaults it to 32,
# which silently constructs a 29M-parameter model that crashes at the first
# forward pass with 6 attention heads. Plain MHA matches the spec's counts.
TINY_CONFIGS = {
    "default": dict(hidden_size=384, num_hidden_layers=8, num_attention_heads=6,
                    num_key_value_heads=6, intermediate_size=1024),
    "smoke": dict(hidden_size=256, num_hidden_layers=6, num_attention_heads=4,
                  num_key_value_heads=4, intermediate_size=704),
}


def build_tiny_model(name: str, tok):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=len(tok),
        tie_word_embeddings=True,
        max_position_embeddings=2048,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        **TINY_CONFIGS[name],
    )
    model = Qwen2ForCausalLM(cfg)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    return model


def encodable(tok, text: str) -> bool:
    """The dataset build gates typed chars, never the target sentence, so a
    prompt can contain a char the closed vocabulary cannot encode. The
    tokenizers OOV error is a plain Exception from pyo3, not ValueError."""
    try:
        tok(text)
        return True
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    from typeshi.tiny_tokenizer import build_tiny_tokenizer
    from typeshi.train_motor import _detect_backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/motor-tiny"))
    ap.add_argument("--config", default="default", choices=sorted(TINY_CONFIGS))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--limit", type=int, default=None,
                    help="seeded random sample of N examples after the mode "
                         "filter. Random, not head-of-file: train.jsonl groups "
                         "writers, and subsets need full writer breadth")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="checkpoint every N steps; default saves at epoch end "
                         "only. Overnight runs must set this (spec §5.4)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the newest checkpoint inside --out")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    args = ap.parse_args()

    backend = _detect_backend()
    print(f"backend: {backend}")

    tok = build_tiny_tokenizer()
    model = build_tiny_model(args.config, tok)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"config {args.config}: {n_params / 1e6:.2f}M params from scratch, "
          f"vocab {len(tok)}")

    ds = load_dataset("json", data_files=str(args.data), split="train")
    ds = ds.filter(lambda r: "<MODE:T>" in r["prompt"])
    if len(ds) == 0:
        raise SystemExit(
            f"no examples with <MODE:T> in {args.data}; "
            "run scripts/build_dataset.py first"
        )

    before = len(ds)
    ds = ds.filter(lambda r: encodable(tok, r["prompt"]))
    if before - len(ds):
        print(f"dropped {before - len(ds)} examples with unencodable prompts")

    if args.limit is not None and args.limit < len(ds):
        ds = ds.shuffle(seed=args.seed).select(range(args.limit))
        print(f"sampled {len(ds)} examples (seed {args.seed})")

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        processing_class=tok,
        args=SFTConfig(
            output_dir=str(args.out),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.accum,
            learning_rate=args.lr,
            # Ratio, not fixed steps: a 25k-example pilot is ~390 optimizer
            # steps, which a warmup_steps=500 would consume entirely.
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.1,
            max_grad_norm=1.0,
            bf16=backend["bf16"],
            logging_steps=25,
            save_strategy="steps" if args.save_steps else "epoch",
            save_steps=args.save_steps or 500,
            save_total_limit=2,
            seed=args.seed,
            max_length=2048,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))

    # Bind the writer split so a later dataset rebuild cannot swap the
    # held-out writers under this model's eval (same move as train_motor).
    split = args.data.parent / "split.json"
    if split.exists():
        shutil.copy(split, args.out / "split.json")
        print(f"bound {split} to the checkpoint")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_tiny.py -q`
Expected: PASS (5 tests). The param-count tests instantiate real models on CPU —
a few seconds each is normal.

- [ ] **Step 5: Commit**

```bash
git add src/typeshi/train_tiny.py tests/test_train_tiny.py
git commit -m "feat: train_tiny -- from-scratch tiny model on the phase-1 recipe"
```

---

### Task 4: Eval loader fallback + OOV skip (`run_eval.py`)

**Files:**
- Create: `src/typeshi/eval/load.py`
- Modify: `scripts/run_eval.py:159-176` (loading), `scripts/run_eval.py:184-257` (loop + reports)
- Test: `tests/test_eval_load.py`

**Interfaces:**
- Consumes: checkpoint layout from Task 3; `_detect_backend()`'s dict shape (`{"dtype", "device_map", "bf16"}`).
- Produces: `typeshi.eval.load.load_checkpoint_model(checkpoint: Path, backend: dict) -> model` used by `run_eval.py`; new report field `sessions_skipped_unencodable_prompt`.

- [ ] **Step 1: Write the failing routing test**

```python
# tests/test_eval_load.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_load.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'typeshi.eval.load'`

- [ ] **Step 3: Implement the loader**

```python
# src/typeshi/eval/load.py
"""Loads an eval checkpoint whether it is a PEFT adapter or a plain model."""

from __future__ import annotations

from pathlib import Path


def load_checkpoint_model(checkpoint: Path, backend: dict):
    """PEFT checkpoints (Phase 1) and plain directories (tiny PoC) both load.

    AutoPeftModelForCausalLM requires adapter_config.json, so its presence is
    the routing signal. `backend` is select_backend()'s dict.
    """
    import torch

    dtype = getattr(torch, backend["dtype"])
    if (checkpoint / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(
            checkpoint, dtype=dtype, device_map=backend["device_map"]
        )
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=dtype, device_map=backend["device_map"]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_load.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Wire `run_eval.py` to it**

In `scripts/run_eval.py` replace the loading block (currently `import torch` /
`from peft import AutoPeftModelForCausalLM` / `AutoPeftModelForCausalLM.from_pretrained(...)`):

```python
    import torch
    from transformers import AutoTokenizer

    from typeshi.eval.load import load_checkpoint_model
    from typeshi.train_motor import _detect_backend

    # Same placement rules as training: device_map="auto" on MPS aborts for
    # hybrid-attention architectures, so load on CPU there and move after.
    backend = _detect_backend()
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = load_checkpoint_model(args.checkpoint, backend)
```

(The `if backend["device_map"] is None and torch.backends.mps.is_available():`
move-to-MPS lines and `model.eval()` stay exactly as they are.)

- [ ] **Step 6: Add the OOV skip to the loop**

In `main()`, add a counter next to the others (`skipped_unencodable = 0` beside
`skipped_train_writer = 0`), import `build_prompt` (`from typeshi.dataset import
build_prompt` at the top with the other typeshi imports), and insert between
`labels = compute_labels(events, target)` and `attempts += 1`:

```python
        try:
            # The closed char vocabulary raises on prompts the build never
            # gated (it checks typed chars, not the target sentence). A
            # marker-containing target lands here too instead of being
            # miscounted as a malformed *generation*.
            tok(build_prompt(target, labels, "transcription"))
        except Exception:  # noqa: BLE001 - pyo3 raises plain Exception
            skipped_unencodable += 1
            continue
```

Add `"sessions_skipped_unencodable_prompt": skipped_unencodable,` to **both**
report dicts (the zero-valid one and the full one), next to
`"sessions_skipped_not_held_out"`.

- [ ] **Step 7: Full offline test suite + eval smoke-parse**

Run: `uv run pytest -q` — Expected: PASS (network/slow skip).
Run: `uv run python -c "import ast; ast.parse(open('scripts/run_eval.py').read())"` — Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add src/typeshi/eval/load.py tests/test_eval_load.py scripts/run_eval.py
git commit -m "feat: eval loads plain checkpoints and skips unencodable prompts"
```

---

### Task 5: Fixture end-to-end micro-test (slow-marked)

Trains the smoke config to overfit one fixture session, reloads the checkpoint
from disk **exactly the way `run_eval.py` does**, and asserts constrained
generation terminates via EOS and deserializes.

**Files:**
- Modify: `pyproject.toml` (register `slow` marker), `tests/conftest.py` (gate it)
- Test: `tests/test_tiny_e2e.py`

**Interfaces:**
- Consumes: `train_tiny.main()` CLI (Task 3), the Task-2 mask behavior, `aalto.iter_sessions`, `build_examples`, `generate()`.
- Produces: the executable proof used as Build-order milestone 3; no code consumed later.

- [ ] **Step 1: Register the `slow` marker**

In `pyproject.toml`, extend the markers list:

```toml
markers = [
    "network: needs to download a model; set TYPESHI_NETWORK_TESTS=1 to run",
    "slow: trains for minutes; set TYPESHI_SLOW_TESTS=1 to run",
]
```

In `tests/conftest.py`, add beside `RUN_NETWORK`:

```python
RUN_SLOW = os.environ.get("TYPESHI_SLOW_TESTS") == "1"
```

and extend `pytest_collection_modifyitems` to also skip `slow`:

```python
def pytest_collection_modifyitems(config, items):
    skips = {}
    if not RUN_NETWORK:
        skips["network"] = pytest.mark.skip(
            reason="needs network; set TYPESHI_NETWORK_TESTS=1 to run")
    if not RUN_SLOW:
        skips["slow"] = pytest.mark.skip(
            reason="trains for minutes; set TYPESHI_SLOW_TESTS=1 to run")
    for item in items:
        for name, mark in skips.items():
            if name in item.keywords:
                item.add_marker(mark)
```

- [ ] **Step 2: Write the e2e test**

```python
# tests/test_tiny_e2e.py
"""Fixture-scale proof of the whole tiny loop: train -> save -> reload from
disk the way run_eval does -> constrained generation -> EOS before budget ->
deserialize. Overfits ONE short session on purpose: a memorized model must
emit EOS at the stream's end, so a hang here means the termination path is
broken (the exact failure the EOS-in-gap fix exists to prevent)."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

from typeshi.adapters import aalto
from typeshi.dataset import build_examples
from typeshi.labels import compute_labels

FIXTURE = Path(__file__).parent / "fixtures" / "aalto_sample.txt"


@pytest.mark.slow
def test_train_reload_generate_terminate(tmp_path, monkeypatch):
    sessions = list(aalto.iter_sessions(FIXTURE))
    assert sessions, "fixture parsed to zero sessions"
    _, target, events = sessions[0]
    labels = compute_labels(events, target)
    examples = build_examples(target, events, labels, "transcription")

    train_file = tmp_path / "train.jsonl"
    with train_file.open("w") as f:
        for _ in range(800):  # ~100 optimizer steps at batch 8
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

    out = tmp_path / "ckpt"
    from typeshi import train_tiny

    monkeypatch.setattr(sys, "argv", [
        "train_tiny", "--data", str(train_file), "--out", str(out),
        "--config", "smoke", "--epochs", "1", "--batch", "8", "--accum", "1",
        "--lr", "1e-2",
    ])
    train_tiny.main()

    # Reload EXACTLY as run_eval.py does: from disk, via the auto classes.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    model.eval()

    from typeshi.generate import generate

    budget = 4 * len(target) + 64
    gen = generate(model, tok, target, labels, mode="transcription",
                   temperature=0.7, max_new_tokens=budget, seed=0)
    assert gen, "constrained generation produced no events"
    # Termination proof: generation stops only on EOS or budget exhaustion.
    # A full-budget stream is ~budget/2 events; well under that means EOS.
    assert 2 * len(gen) + 2 < budget, "burned the whole budget -- EOS never came"
```

- [ ] **Step 3: Run it (slow gate on)**

Run: `TYPESHI_SLOW_TESTS=1 uv run pytest tests/test_tiny_e2e.py -q -x`
Expected: PASS in single-digit minutes. If the EOS assertion fails, do **not**
weaken it — that assertion existing is the point; debug the termination path
(Task 2's mask, `generation_config` ids) instead.

- [ ] **Step 4: Confirm it skips by default**

Run: `uv run pytest tests/test_tiny_e2e.py -q`
Expected: `1 skipped`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/test_tiny_e2e.py
git commit -m "test: fixture e2e -- tiny model trains, reloads, terminates"
```

---

### Task 6: Full dataset rebuild

The current `data/processed/train.jsonl` is the 17,714-example overnight subset.
The pilot and full run sample from the complete transcription export.

**Files:**
- No code changes; regenerates `data/processed/{train,test}.jsonl` + `split.json`.

**Interfaces:**
- Consumes: `scripts/build_dataset.py` defaults (`--aalto data/raw/aalto`, `--out data/processed`).
- Produces: full `train.jsonl` (~2.0M examples incl. composition; ~1.99M transcription) used by Tasks 7–10.

- [ ] **Step 1: Rebuild (about 25 minutes, measured at 115 files/sec)**

```bash
caffeinate -dim uv run python scripts/build_dataset.py 2>&1 | tee logs/build_full.log
```

- [ ] **Step 2: Verify the scale**

```bash
wc -l data/processed/train.jsonl
uv run python -c "
import json
n = sum(1 for l in open('data/processed/train.jsonl') if '<MODE:T>' in json.loads(l)['prompt'])
print('transcription examples:', n)"
```

Expected: train.jsonl on the order of 2M lines; transcription count ≈ 1.99M
(the exact number goes in the results doc).

- [ ] **Step 3: Rebuild the held-out symlink dir (eval speed, per gpu-handoff.md)**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

split = json.loads(Path("data/processed/split.json").read_text())
writers = {w.split(":", 1)[1] for w in split["test_writers"] if w.startswith("aalto:")}
outdir = Path("data/processed/heldout_aalto")
outdir.mkdir(exist_ok=True)
for old in outdir.glob("*_keystrokes.txt"):
    old.unlink()
n = 0
for f in Path("data/raw/aalto").rglob("*_keystrokes.txt"):
    if f.stem.split("_")[0] in writers:
        (outdir / f.name).symlink_to(f.resolve())
        n += 1
print(f"linked {n} held-out logs")
EOF
```

Expected: a few thousand linked logs (10% of physical-keyboard writers).

- [ ] **Step 4: Commit nothing** — `data/` is untracked; just record the counts for the results doc.

---

### Task 7: Harness-ceiling control (spec §5a)

Measures whether serialize→deserialize quantization **alone** is discriminable.
Outside [0.40, 0.55], `pass_model` is unreachable for any generator and the
featurization must be fixed before any realism number is interpreted.

**Files:**
- Create: `scripts/harness_control.py`

**Interfaces:**
- Consumes: `aalto.iter_sessions`, `train_discriminator(real, fake, paired=True, count_features=…)`, `serialize`/`deserialize`, `split.json` from Task 6.
- Produces: `harness_control.json` with `ceiling_full_features` / `ceiling_timing_only`, quoted in Tasks 9–11.

- [ ] **Step 1: Write the script**

```python
# scripts/harness_control.py
"""One-off harness-ceiling control: real vs serialize->deserialize(real).

The realism gates compare generations carrying BIN-CENTER timings (they exit
deserialize()) against real sessions carrying raw milliseconds. Whether that
asymmetry alone is discriminable has never been measured -- the documented
0.085-on-exact-copies calibration used raw copies, not round-tripped ones.
If this lands outside ~[0.40, 0.55], pass_model measures the harness, not
the generator, and featurization must change before ANY realism number (tiny
or 7B) is interpreted. Spec 2026-08-10, §5a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typeshi.adapters import aalto
from typeshi.eval.discriminator import train_discriminator
from typeshi.serialize import deserialize, serialize, unsupported_chars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", type=Path,
                    default=Path("data/processed/heldout_aalto"))
    ap.add_argument("--split", type=Path,
                    default=Path("data/processed/split.json"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("harness_control.json"))
    args = ap.parse_args()

    payload = json.loads(args.split.read_text())
    test_writers = set(payload["test_writers"])

    real, roundtripped = [], []
    for writer, _target, events in aalto.iter_sessions(args.held_out):
        if len(real) >= args.n:
            break
        if f"aalto:{writer}" not in test_writers:
            continue
        if not events or unsupported_chars(events):
            continue
        real.append(events)
        # The same session through the token format: identical chars and
        # event counts, timings snapped to bin centers. Any accuracy above
        # chance here is pure quantization signal.
        roundtripped.append(deserialize(serialize(events)))

    _, acc_full = train_discriminator(real, roundtripped, paired=True)
    _, acc_timing = train_discriminator(
        real, roundtripped, paired=True, count_features=False
    )
    report = {
        "sessions": len(real),
        "ceiling_full_features": acc_full,
        "ceiling_timing_only": acc_timing,
        "band": [0.40, 0.55],
        "interpretation": (
            "outside the band, pass_model measures harness quantization, "
            "not the generator -- fix featurization before reading realism"
        ),
    }
    text = json.dumps(report, indent=2)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/harness_control.py --n 200 --out harness_control.json
```

Expected: a JSON report with `sessions: 200`. Record both ceilings.

- [ ] **Step 3: Interpret against the band**

- Both ceilings in [0.40, 0.55] → the realism gates are meaningful; proceed.
- Either outside → **stop and surface to the user before the overnight run**:
  the spec's named fixes are quantizing real timings through the same bins
  before featurizing, or jittering fakes within-bin; that change is its own
  reviewed task, not an inline patch.

- [ ] **Step 4: Commit**

```bash
git add scripts/harness_control.py
git commit -m "feat: harness-ceiling control -- real vs round-tripped real"
```

---

### Task 8: Throughput smoke (`--limit 1000`)

**Files:**
- No new code. Produces `checkpoints/tiny-throughput-smoke/` and a measured ex/s.

**Interfaces:**
- Consumes: `train_tiny` CLI, full dataset (Task 6).
- Produces: the ladder decision (spec §5.2) that sets Task 10's shape.

- [ ] **Step 1: Run the smoke**

```bash
caffeinate -dim uv run python -m typeshi.train_tiny \
  --config smoke --limit 1000 --epochs 1 \
  --out checkpoints/tiny-throughput-smoke 2>&1 | tee logs/tiny_smoke.log
```

- [ ] **Step 2: Record the numbers**

From the end-of-training metrics in the log, record:
`train_samples_per_second` (the ex/s figure), total wall clock, and how long
the dataset filter/tokenize-map took before step 1 (the gap between launch
and the first `logging_steps` line), extrapolated ×2,000 to the full corpus.

- [ ] **Step 3: Apply the ladder (spec §5.2)**

With `H = 1_989_167 / ex_s / 3600` (recompute with the exact Task 6 count):

- `H ≤ 12` (ex/s ≥ ~46) → Task 10 runs the full epoch in one night.
- `12 < H ≤ 24` → Task 10 becomes half-corpus (`--limit` at half the
  transcription count), second night resumes with `--resume`.
- `H > 24` → **stop; surface to the user** (MLX port or multi-night are the
  spec's named options — user decision, not an inline choice).

Also note the map extrapolation: if tokenize-map × 2,000 exceeds ~1 h, plan the
full run with the map cost included in the night, or pre-tokenize (a decision to
surface with the ladder result, not improvise).

- [ ] **Step 4: Sanity-check the loss**

`grep "'loss'" logs/tiny_smoke.log | head -20` — Expected: loss falling from
its ~9.5 random-init start (ln 12,909) within the first hundred steps.

---

### Task 9: Two-point pilot (25k / 100k)

**Files:**
- No new code. Produces two checkpoints + two eval reports.

**Interfaces:**
- Consumes: `train_tiny` CLI, patched `run_eval.py` (Task 4), symlink dir (Task 6).
- Produces: the go/stop decision for Task 10, per spec §5.3 bands.

- [ ] **Step 1: Train both pilots (19M config)**

```bash
caffeinate -dim uv run python -m typeshi.train_tiny \
  --config default --limit 25000 --epochs 1 --save-steps 500 \
  --out checkpoints/tiny-pilot-25k 2>&1 | tee logs/tiny_pilot_25k.log

caffeinate -dim uv run python -m typeshi.train_tiny \
  --config default --limit 100000 --epochs 1 --save-steps 500 \
  --out checkpoints/tiny-pilot-100k 2>&1 | tee logs/tiny_pilot_100k.log
```

- [ ] **Step 2: Eval both**

```bash
uv run python scripts/run_eval.py --checkpoint checkpoints/tiny-pilot-25k \
  --held-out data/processed/heldout_aalto --n 50 --out eval_tiny_pilot25k.json
uv run python scripts/run_eval.py --checkpoint checkpoints/tiny-pilot-100k \
  --held-out data/processed/heldout_aalto --n 50 --out eval_tiny_pilot100k.json
```

Record `generation_success_rate` from both (±6 points at 150 attempts — treat
differences inside that as noise).

- [ ] **Step 3: Decide per the spec §5.3 bands**

- **Proceed** to Task 10 if 100k-validity ≥ 0.40, **or** ≥ 2× the 25k-validity.
- **Middle band:** run the one predeclared lever — a second epoch on the 100k
  subset (`--epochs 2`, fresh `--out checkpoints/tiny-pilot-100k-e2`), re-eval,
  then force the decision: proceed on the same bands or stop. No further levers.
- **Stop** if 100k-validity < 0.05 after that lever → surface to the user; the
  next step is the encoder-decoder fallback, which gets its own design pass
  (spec §5), not an improvised variant.

---

### Task 10: Full overnight run — **user-gated**

**Do not launch this unattended without explicit user go-ahead** — it occupies
the machine for the night; the user may be using it.

**Files:**
- No new code. Produces `checkpoints/motor-tiny/` + `eval_tiny_full.json`.

- [ ] **Step 1: Confirm with the user** — ladder shape from Task 8, pilot numbers
from Task 9, tonight vs. later.

- [ ] **Step 2: Launch per the ladder**

Full-epoch shape (adjust `--limit` per the Task 8 ladder if half-corpus;
`--save-steps` at ~30–60 min of measured throughput, e.g. ex/s × 2700 / 64):

```bash
caffeinate -dim uv run python -m typeshi.train_tiny \
  --config default --epochs 1 --save-steps <computed> \
  --out checkpoints/motor-tiny 2>&1 | tee logs/tiny_full.log
```

If it dies mid-night: relaunch the identical command plus `--resume` — cost is
one save interval, not the night.

- [ ] **Step 3: Final eval**

```bash
uv run python scripts/run_eval.py --checkpoint checkpoints/motor-tiny \
  --held-out data/processed/heldout_aalto --n 200 --out eval_tiny_full.json
```

- [ ] **Step 4: Read the gates**

Hard bar: `pass_generation_validity`, `pass_discriminator_has_teeth`,
`pass_control_near_chance`. Stretch: `pass_model`, `pass_serial_dependence_teeth`
— read the stretch numbers **only next to Task 7's measured ceiling**. Also
compare `discriminator_accuracy_vs_model_timing_only` vs the full number (a big
gap = caught on length, not timing).

---

### Task 11: Results doc

**Files:**
- Create: `docs/results-tiny-poc.md`

- [ ] **Step 1: Write it**, following `docs/results-08b-shakedown.md`'s shape:
training table (config, params, examples, epochs, wall clock, final loss); the
harness-ceiling numbers from Task 7 (full + timing-only, with the band
interpretation); pilot table (25k/100k validity, the band hit); the full-run
five-gate table; what the result does and does not prove (copy the spec §1
calibration — hard bar ≠ timing soundness; GPU residual risks restated:
BPE-copying first); surprises found along the way.

- [ ] **Step 2: Commit**

```bash
git add docs/results-tiny-poc.md
git commit -m "docs: tiny motor PoC results"
```

---

## Execution notes

- Task order is strict except Tasks 6–7 (rebuild + harness control) which may
  run while code tasks are reviewed; Task 7 needs Task 6's split.
- Tasks 8–10 are runs, not code: their "tests" are the recorded numbers and the
  predeclared decision bands. Never improvise past a stop condition — stop
  conditions surface to the user.
- Full offline suite after every code task: `uv run pytest -q`.
