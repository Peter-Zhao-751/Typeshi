"""Builds an MLX 4-bit base model that knows the event token grammar.

`mlx-lm` loads whatever tokenizer ships with the model repo, and no stock
tokenizer has our ~380 event tokens. Training against one costs 4.13x more
tokens per example (measured: 603 vs 146), spreads each event's timing across
several BPE fragments, and truncates long sessions at the sequence limit.

This script does the tokenizer surgery *before* MLX ever sees the model:

    full-precision HF model
      -> prepare_tokenizer()  (add the event tokens)
      -> resize_token_embeddings()
      -> save locally
      -> mlx_lm convert -q    (4-bit)

Run once, then point `mlx_lm lora --model` at the output directory.

    uv run python scripts/prepare_mlx_model.py --out models/llama31-8b-typeshi-mlx

Needs roughly 16 GB of download, 16 GB for the intermediate, and 5 GB for the
converted model.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from typeshi import config
from typeshi.serialize import special_tokens
from typeshi.train_motor import initialize_new_token_embeddings, prepare_tokenizer


def extend_and_save(base_model: str, dest: Path, dtype: str) -> int:
    """Saves `base_model` with the event tokens registered. Returns vocab size."""
    import torch
    from transformers import AutoModelForCausalLM

    print(f"loading {base_model} in {dtype} (this downloads ~16 GB the first time)")
    # Plain CPU load: bfloat16 with device_map="auto" segfaults on MPS, and we
    # only need the weights on CPU long enough to resize and write them out.
    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=getattr(torch, dtype)
    )

    tok = prepare_tokenizer(base_model)
    before = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tok))
    after = model.get_input_embeddings().weight.shape[0]
    print(f"embeddings resized {before} -> {after} for {len(tok)} tokenizer entries")

    # This matters far more here than on the HF path. `mlx_lm lora` trains only
    # the attention adapters and never touches the embedding table, so whatever
    # these rows hold at conversion time is what they hold forever. Resizing
    # alone leaves every event token with essentially the same vector.
    seeded = initialize_new_token_embeddings(model, base_model)
    print(f"seeded {seeded} event-token embeddings from their sub-word pieces")

    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    print(f"wrote extended model to {dest}")
    return len(tok)


def convert_to_mlx(src: Path, dest: Path, quantize: bool, bits: int) -> None:
    cmd = [sys.executable, "-m", "mlx_lm", "convert",
           "--hf-path", str(src), "--mlx-path", str(dest)]
    if quantize:
        cmd += ["-q", "--q-bits", str(bits)]
    print(f"running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def verify(dest: Path) -> None:
    """Confirms the converted model tokenizes the grammar one token per event."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(dest)
    probes = ["<DT:0>", "<DT:127>", "<a:51>", "<SPC:17>", "<BKSP:5>",
              "<MODE:T>", "<WPM:18>", "<TARGET>", "<PROCESS>"]
    bad = [p for p in probes if len(tok.tokenize(p)) != 1]
    if bad:
        raise SystemExit(f"FAILED: these did not survive as single tokens: {bad}")

    sample = "<H:51><DT:49><e:51><DT:50><SPC:51><DT:50>"
    pieces = tok.tokenize(sample)
    print(f"verified: {len(probes)} probe tokens are single tokens")
    print(f"  {sample}\n  -> {pieces}")
    if len(pieces) != 6:
        raise SystemExit(f"FAILED: expected 6 tokens for the sample, got {len(pieces)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.BASE_MODEL)
    ap.add_argument("--out", type=Path, default=Path("models/base-typeshi-mlx"))
    ap.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="where to write the extended fp16 model (default: <out>-hf)",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--q-bits", type=int, default=4)
    ap.add_argument("--no-quantize", action="store_true")
    ap.add_argument(
        "--keep-staging",
        action="store_true",
        help="keep the ~16 GB intermediate instead of deleting it",
    )
    args = ap.parse_args()

    staging = args.staging or args.out.with_name(args.out.name + "-hf")

    n_tokens = extend_and_save(args.base, staging, args.dtype)
    convert_to_mlx(staging, args.out, not args.no_quantize, args.q_bits)
    verify(args.out)

    if not args.keep_staging:
        shutil.rmtree(staging, ignore_errors=True)
        print(f"removed staging directory {staging}")

    print(
        f"\ndone. {len(special_tokens())} grammar entries, {n_tokens} tokenizer "
        f"entries.\nTrain with:\n"
        f"  python -m mlx_lm lora --model {args.out} --train \\\n"
        f"    --data <dir-with-train-and-valid-jsonl> \\\n"
        f"    --batch-size 1 --num-layers 8 --iters 200 --max-seq-length 1024 \\\n"
        f"    --adapter-path checkpoints/motor-mlx"
    )


if __name__ == "__main__":
    main()
