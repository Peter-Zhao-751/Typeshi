# Training on Apple Silicon

Measured on an **M5 Pro, 15 cores, 48 GB unified memory**, torch 2.13,
transformers 5.14, Python 3.14.

Short answer: yes, the GPU is usable, but **not through PyTorch for an 8B
model**. Use MLX for that. PyTorch/MPS is fine for smoke tests on small models.

## PyTorch MPS

MPS (Metal Performance Shaders) is the Apple equivalent of the CUDA backend and
is available out of the box — `torch.backends.mps.is_available()` is `True`.
The constraint is precision, not availability:

| Configuration | Result |
|---|---|
| `dtype=bfloat16` alone | loads fine |
| `device_map="auto"` alone (lands on MPS) | loads fine |
| `dtype=bfloat16` + `device_map="auto"` | **segfault (exit 139)** while loading |
| `dtype=float16` + `.to("mps")` | **hangs indefinitely** |
| `dtype=float32` | works — 100 training steps verified |

`select_backend()` in `src/typeshi/train_motor.py` encodes this: CUDA keeps
bf16 as the plan intends, MPS falls back to fp32, and the combination that
crashes is never emitted. `tests/test_train_motor.py` pins that behaviour.

### Why fp32 rules out an 8B here

fp32 doubles the weight footprint against the plan's bf16 assumption:

| Model | bf16 weights | fp32 weights |
|---|---|---|
| 8B | ~16 GB | **~32 GB** |
| 1–3B | ~2–6 GB | ~4–12 GB |

32 GB of a 48 GB machine, before activations and before the OS takes its share,
is not a workable training configuration. Small models are fine.

## MLX — the Apple-native route

[MLX](https://github.com/ml-explore/mlx) is Apple's own array framework:
Metal-native, designed around unified memory, with `mlx-lm` providing LoRA and
QLoRA fine-tuning directly. A 4-bit 8B is roughly **4.5 GB** rather than 32 GB.

```bash
uv pip install mlx mlx-lm
```

`mlx-lm` reads `{"prompt": ..., "completion": ...}` JSONL natively, which is
exactly what `scripts/build_dataset.py` already emits. It expects a directory
containing `train.jsonl` and `valid.jsonl`:

```bash
python -m mlx_lm lora \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --train --data <dir-with-train-and-valid-jsonl> \
  --batch-size 1 --num-layers 8 --iters 200 --max-seq-length 1024 \
  --adapter-path checkpoints/motor-mlx
```

### Caveat: adapters are not interchangeable

`mlx-lm` writes MLX-format adapters, which `scripts/run_eval.py` cannot load —
it uses `AutoPeftModelForCausalLM`. To evaluate an MLX-trained adapter, fuse it
back into a Hugging Face checkpoint first:

```bash
python -m mlx_lm fuse \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path checkpoints/motor-mlx \
  --save-path checkpoints/motor-fused
```

Also note MLX has no equivalent of the tokenizer surgery in
`prepare_tokenizer()`. The event grammar's ~380 special tokens
(`<DT:k>`, `<KEY:c>`, `<HOLD:k>`) are added to the HF tokenizer and the
embedding matrix is resized to match. Reproducing that under MLX is unsolved
here — without it, every event token fragments into several BPE pieces, which
inflates sequence length several-fold and weakens the timing signal the model
is supposed to learn.

## Base model access

`config.BASE_MODEL` is `meta-llama/Meta-Llama-3.1-8B-Instruct`, which is
gated (`gated=manual`). Access is granted on this machine's HF token. Ungated
alternatives that need no approval:

- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (MLX, 4-bit)
- `Qwen/Qwen2.5-7B-Instruct` (the plan allows Llama *or* Qwen)

## Recommendation

| Goal | Route |
|---|---|
| Verify the pipeline runs | PyTorch/MPS, tiny model, `--limit-*` data |
| Iterate on an 8B locally | MLX + 4-bit, accepting the tokenizer caveat |
| The real 2M-example run | Rented CUDA GPU — bf16, as the plan intends |

The full export is 2,018,334 examples. At batch 4 × accum 8 that is ~63k
optimizer steps for one epoch, which is a rented-GPU job regardless of how well
the Mac path works.
