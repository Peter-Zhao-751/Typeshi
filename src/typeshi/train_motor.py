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
        return {"dtype": "float32", "device_map": "auto", "bf16": False}
    return {"dtype": "float32", "device_map": None, "bf16": False}


def _detect_backend() -> dict:
    import torch

    return select_backend(
        has_cuda=torch.cuda.is_available(),
        has_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        has_mps=torch.backends.mps.is_available(),
    )


def build_peft_config():
    from peft import LoraConfig

    return LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    model = get_peft_model(model, build_peft_config())
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
