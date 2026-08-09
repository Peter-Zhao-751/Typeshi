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

### Measured throughput

4-bit 8B Llama 3.1, real transcription examples, `--max-seq-length 1024`:

| batch | LoRA layers | tokens/sec | peak memory |
|---|---|---|---|
| **1** | **8** | **554** | **8.5 GB** |
| 4 | 8 | ~370 | 19.1 GB |
| 4 | 16 | 270 | 29.2 GB |

**Batch 1 is the fastest configuration.** Training here is memory-bandwidth
bound, so larger batches cost more than they buy — batch 4 with 16 layers is
half the throughput of batch 1 with 8, for 3.4× the memory. Peak memory at the
fast setting is 8.5 GB of 48 GB, so the headroom is real but not worth spending
on batch size.

Training does converge: val loss fell 1.346 → 0.563 over 20 iterations.

### Caveat: the event tokens fragment, 4.13x

This is the important one. `prepare_tokenizer()` registers the ~380 event
tokens so each event is exactly one token. `mlx-lm` loads the stock tokenizer
from the model repo and has no equivalent step, so the grammar shatters:

```
<DT:0><KEY:W><HOLD:52>
  stock : ['<','DT',':','0','><','KEY',':','W','><','H','OLD',':','52','><']
  ours  : ['<DT:0>', '<KEY:W>', '<HOLD:52>']
```

Measured over 50 real completions: **603 tokens/example stock vs 146 extended
— 4.13x inflation.** Consequences:

- Effective throughput drops ~4x, since most tokens are punctuation fragments.
- At `--max-seq-length 1024` sequences get truncated (observed: 1161 → 1024),
  so the model sees only part of each session.
- The timing signal is spread across several tokens instead of one, which is
  precisely the representation the plan chose bins to avoid.

**The MLX numbers above were measured under this penalty.** They are honest
throughput figures for the naive setup, not for the intended representation.

Fixing it means building the extended model locally before MLX ever sees it:

1. Load the full-precision 8B in transformers, run `prepare_tokenizer()`,
   `resize_token_embeddings()`, and save the pair to a local directory.
2. `python -m mlx_lm convert --hf-path <local-dir> -q` to quantize to 4-bit.
3. Train against that converted local model.

This is untested here. It needs the ~16 GB full-precision download plus disk
for the conversion.

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
| Iterate on an 8B locally | MLX + 4-bit, after fixing the tokenizer |
| The real 2M-example run | Rented CUDA GPU — bf16, as the plan intends |

### Why the full run does not belong here

The export is 2,018,334 examples. At ~200 tokens each with the extended
tokenizer that is ~400M tokens per epoch; at the measured 554 tokens/sec:

| Scope | Tokens | Wall clock at 554 tok/s |
|---|---|---|
| Full epoch, extended tokenizer | ~400M | **~8 days** |
| Full epoch, fragmented (today) | ~1.4B | **~29 days** |
| 50k-example subset, extended | ~10M | ~5 hours |

So the Mac is genuinely useful for a subset run overnight, and genuinely
unsuitable for the full corpus. Rent the GPU for the real thing.
