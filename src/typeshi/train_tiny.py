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
