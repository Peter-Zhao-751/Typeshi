"""Phase 1: LoRA fine-tune on transcription data to learn motor timing."""

from __future__ import annotations

import argparse
from pathlib import Path

from typeshi import config
from typeshi.serialize import special_tokens


def prepare_tokenizer(base_model: str = config.BASE_MODEL):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    # Prefix-only entries ("<CUR:", "<SELDEL:") carry variable integers, so only
    # whole tokens are registered; the integers tokenize as ordinary digits.
    whole = [t for t in special_tokens() if t.endswith(">")]
    tok.add_tokens(whole, special_tokens=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def select_backend(has_cuda: bool, has_bf16: bool, has_mps: bool) -> dict:
    """Chooses dtype / placement / mixed precision for the available hardware.

    bfloat16 weights combined with `device_map="auto"` segfault on Apple
    Silicon (torch 2.13, MPS): either alone is fine, together they crash while
    loading. CUDA is the intended target and keeps bf16; everything else falls
    back to fp32 so the same script still runs for smoke tests.
    """
    if has_cuda and has_bf16:
        return {"dtype": "bfloat16", "device_map": "auto", "bf16": True}
    if has_cuda:
        return {"dtype": "float16", "device_map": "auto", "bf16": False}
    if has_mps:
        # fp32 is the only precision that works reliably here. Measured on an
        # M5 Pro with torch 2.13: bfloat16 + device_map="auto" segfaults while
        # loading (exit 139), and float16 hangs indefinitely. fp32 doubles the
        # memory an 8B needs to ~32 GB, so MPS is for smoke tests on small
        # models -- see docs/training-on-apple-silicon.md for the MLX route.
        return {"dtype": "float32", "device_map": "auto", "bf16": False}
    return {"dtype": "float32", "device_map": None, "bf16": False}


def _detect_backend() -> dict:
    import torch

    return select_backend(
        has_cuda=torch.cuda.is_available(),
        has_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        has_mps=torch.backends.mps.is_available(),
    )


def initialize_new_token_embeddings(model, base_model: str) -> int:
    """Seeds each event token's embedding from the pieces it used to split into.

    `resize_token_embeddings` draws new rows from a distribution fitted to the
    old ones, which is random noise as far as meaning goes. That throws away
    the structure the grammar depends on: <DT:50> and <DT:51> are adjacent time
    bins and should start close together, and every <KEY:x> should start near
    the ordinary character x.

    Averaging the base tokenizer's pieces for each token recovers exactly that
    -- "<DT:50>" and "<DT:51>" share most of their pieces, so their seeds land
    near each other. Returns the number of rows initialised.
    """
    import torch
    from transformers import AutoTokenizer

    base_tok = AutoTokenizer.from_pretrained(base_model)
    ext_tok = prepare_tokenizer(base_model)

    embed_in = model.get_input_embeddings().weight
    head = model.get_output_embeddings()
    embed_out = head.weight if head is not None else None

    written = 0
    with torch.no_grad():
        for token in special_tokens():
            if not token.endswith(">"):
                continue  # prefix-only entries were never registered
            new_id = ext_tok.convert_tokens_to_ids(token)
            if new_id is None or new_id >= embed_in.shape[0]:
                continue
            piece_ids = base_tok(token, add_special_tokens=False)["input_ids"]
            piece_ids = [i for i in piece_ids if i < embed_in.shape[0]]
            if not piece_ids:
                continue
            index = torch.tensor(piece_ids, device=embed_in.device)
            embed_in[new_id] = embed_in.index_select(0, index).mean(0)
            if embed_out is not None and embed_out is not embed_in:
                embed_out[new_id] = embed_out.index_select(0, index).mean(0)
            written += 1
    return written


def embedding_modules_to_save(tied: bool) -> list[str]:
    """Which embedding modules LoRA should train alongside the adapters.

    When a model ties its input and output embeddings (Qwen does; Llama 3.1
    does not), naming both makes peft warn and complicates merging, because
    they are one tensor. Saving the input side alone covers both.
    """
    return ["embed_tokens"] if tied else ["embed_tokens", "lm_head"]


def build_peft_config(train_embeddings: bool = True, tied_embeddings: bool = False):
    """LoRA config for the motor fine-tune.

    `train_embeddings` adds the embedding matrix and output head to the
    trainable set. It defaults on because the event tokens are new: without it
    every <DT:k>/<KEY:c>/<HOLD:k> keeps whatever vector resizing happened to
    give it, and the output head can never learn to emit them properly. LoRA on
    the attention projections alone cannot compensate -- it never touches the
    embedding table.

    The cost is real. On an 8B with a 128k vocab this adds roughly 1B trainable
    parameters (~13 GB with AdamW states) on top of LoRA's ~40M. Turn it off
    only if the GPU cannot hold that, and expect worse timing fidelity.
    """
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=(
            embedding_modules_to_save(tied_embeddings) if train_embeddings else None
        ),
    )


def main() -> None:
    import torch
    from datasets import load_dataset
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("checkpoints/motor"))
    ap.add_argument("--base", default=config.BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--mode", default="transcription",
                    help="filter examples by MODE= in the prompt")
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument(
        "--freeze-embeddings",
        action="store_true",
        help="do not train the embedding table or output head; saves ~13 GB at "
             "the cost of leaving the new event tokens untrained",
    )
    args = ap.parse_args()

    backend = _detect_backend()
    print(f"backend: {backend}")

    tok = prepare_tokenizer(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=getattr(torch, backend["dtype"]),
        device_map=backend["device_map"],
    )
    model.resize_token_embeddings(len(tok))
    seeded = initialize_new_token_embeddings(model, args.base)
    print(f"seeded {seeded} event-token embeddings from their sub-word pieces")
    model = get_peft_model(
        model,
        build_peft_config(
            train_embeddings=not args.freeze_embeddings,
            tied_embeddings=bool(getattr(model.config, "tie_word_embeddings", False)),
        ),
    )
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(args.data), split="train")
    ds = ds.filter(lambda r: f"MODE={args.mode}" in r["prompt"])
    if len(ds) == 0:
        raise SystemExit(
            f"no examples with MODE={args.mode} in {args.data}; "
            "run scripts/build_dataset.py first"
        )
    ds = ds.map(lambda r: {"text": r["prompt"] + r["completion"] + tok.eos_token})

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
            bf16=backend["bf16"],
            logging_steps=25,
            save_strategy="epoch",
            seed=args.seed,
            max_length=2048,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))


if __name__ == "__main__":
    main()
