import pytest

transformers = pytest.importorskip("transformers")

from typeshi.serialize import special_tokens
from typeshi.train_motor import build_peft_config, prepare_tokenizer

TINY = "hf-internal-testing/tiny-random-gpt2"
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"

# Only the tokenizer tests need to download a model; the rest are pure logic.
needs_network = pytest.mark.network


@needs_network
def test_special_tokens_are_added_to_the_vocabulary():
    tok = prepare_tokenizer(TINY)
    for t in ["<BKSP:0>", "<DT:0>", "<SPC:17>", "<MODE:T>", "<TARGET>"]:
        assert len(tok.tokenize(t)) == 1, f"{t} was split into multiple tokens"


@needs_network
def test_event_tokens_survive_encode_decode():
    tok = prepare_tokenizer(TINY)
    s = "<a:51><DT:49><e:51><DT:50><BKSP:5>"
    assert tok.decode(tok.encode(s), skip_special_tokens=False).replace(" ", "") == s


@needs_network
def test_vocabulary_grew_by_the_expected_amount():
    base = transformers.AutoTokenizer.from_pretrained(TINY)
    tok = prepare_tokenizer(TINY)
    # 2 entries ("<CUR:", "<SELDEL:") are prefixes, not whole tokens
    assert len(tok) >= len(base) + len(special_tokens()) - 2


def test_peft_config_targets_attention_projections():
    cfg = build_peft_config()
    assert cfg.r >= 16
    assert "q_proj" in cfg.target_modules and "v_proj" in cfg.target_modules


# --- backend selection is pure logic, so it needs no model download ---

@pytest.mark.parametrize(
    "cuda,bf16,mps,expect_dtype,expect_bf16",
    [
        (True, True, False, "bfloat16", True),    # the intended GPU target
        # Older CUDA cards fall back to fp32, NOT fp16: loading fp16 weights
        # without a GradScaler is unscaled full-fp16 training, which diverges.
        (True, False, False, "float32", False),
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


def test_half_precision_is_never_emitted_without_mixed_precision():
    """Half-precision weights with bf16=False and no fp16 flag mean training
    with no loss scaling at all. No backend combination may produce that."""
    from typeshi.train_motor import select_backend

    for cuda in (True, False):
        for bf16 in (True, False):
            for mps in (True, False):
                got = select_backend(has_cuda=cuda, has_bf16=bf16, has_mps=mps)
                if got["dtype"] in ("float16", "bfloat16"):
                    assert got["bf16"], f"unscaled half precision from {cuda,bf16,mps}"


def test_tied_models_get_weight_tying_preserved():
    cfg = build_peft_config(train_embeddings=True, tied_embeddings=True)
    assert cfg.ensure_weight_tying
    cfg = build_peft_config(train_embeddings=True, tied_embeddings=False)
    assert not cfg.ensure_weight_tying


def test_embedding_modules_avoid_naming_a_tied_tensor_twice():
    """Tied models share one tensor between input and output embeddings, so
    naming both makes peft warn and complicates merging."""
    from typeshi.train_motor import embedding_modules_to_save

    assert embedding_modules_to_save(tied=True) == ["embed_tokens"]
    assert embedding_modules_to_save(tied=False) == ["embed_tokens", "lm_head"]


def test_new_event_tokens_are_trained_by_default():
    """The event tokens do not exist in any base vocabulary. If the embedding
    table stays frozen they keep whatever resizing gave them for the whole
    run, and the output head can never learn to emit them."""
    cfg = build_peft_config()
    assert cfg.modules_to_save and "embed_tokens" in cfg.modules_to_save


def test_embedding_training_can_be_disabled_for_tight_gpus():
    assert build_peft_config(train_embeddings=False).modules_to_save is None


@needs_network
def test_seeding_gives_adjacent_time_bins_similar_embeddings():
    """<DT:50> and <DT:51> are neighbouring time bins and should start close.

    resize_token_embeddings draws every new row from one fitted distribution,
    which leaves them nearly identical to each other and carrying no ordinal
    structure at all. Seeding from sub-word pieces restores it.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from typeshi.train_motor import initialize_new_token_embeddings

    tok = prepare_tokenizer(TINY_LLAMA)
    model = AutoModelForCausalLM.from_pretrained(TINY_LLAMA, dtype=torch.float32)
    model.resize_token_embeddings(len(tok))
    assert initialize_new_token_embeddings(model, TINY_LLAMA) > 300

    emb = model.get_input_embeddings().weight
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<DT:50>", "<DT:51>", "<DT:120>")}
    sim = lambda a, b: torch.nn.functional.cosine_similarity(  # noqa: E731
        emb[ids[a]][None], emb[ids[b]][None]
    ).item()
    assert sim("<DT:50>", "<DT:51>") > sim("<DT:50>", "<DT:120>")
