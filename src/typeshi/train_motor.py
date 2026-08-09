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

    tok = prepare_tokenizer(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
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
            bf16=True,
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
