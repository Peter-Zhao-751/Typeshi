# Training on Apple Silicon

> **Note:** the measurements in this file were taken under the v1 token
> format (3 tokens/keystroke, 356 special tokens). Format v2
> (`docs/token-format.md`) cuts sequences a further ~35%, so treat the
> throughput and wall-clock figures here as upper bounds. The MLX model
> under `models/` has been rebuilt for v2 (12,810 grammar tokens, verified
> single-token) and trains at ~2.8 examples/sec, 5.8 GB peak.

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

### Fixed: build the model with the grammar baked in

`scripts/prepare_mlx_model.py` does the surgery before MLX sees the model —
extend the tokenizer, seed the new embeddings, resize, convert to 4-bit, and
verify each event survives as one token:

```bash
uv run python scripts/prepare_mlx_model.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --out models/qwen25-7b-typeshi-mlx
```

Roughly 15 GB of download and a few minutes of conversion; the output is 4.0 GB
and the intermediate is deleted unless you pass `--keep-staging`.

Measured before and after, both at batch 1 / 8 layers / seq 1024:

| Configuration | tokens/example | examples/sec | peak memory |
|---|---|---|---|
| Llama-3.1-8B, stock tokenizer | 702 | 0.79 | 8.5 GB |
| **Qwen2.5-7B, extended tokenizer** | **239** | **1.88** | **6.1 GB** |

**2.4x more examples per second and less memory.** Truncation also stops:
sequences no longer exceed the 1024 limit, so the model sees whole sessions
instead of the first quarter. (The two rows are different base models — Llama
is gated — so treat the ratio as indicative, not a controlled A/B.)

### Caveat: MLX cannot train the embeddings

`mlx_lm lora` offers `lora`, `dora`, and `full`, with no way to train the
embedding table alongside adapters. The event tokens are new, so whatever
`prepare_mlx_model.py` seeds at conversion time is **permanent** for the whole
MLX run.

That is why the seeding step exists rather than relying on
`resize_token_embeddings`. Resizing draws every new row from one fitted
distribution, which leaves all 356 tokens nearly identical — measured
`cos(<DT:50>, <DT:120>) = 1.0000`. Seeding from sub-word pieces restores the
ordinal structure the time bins depend on: after it, adjacent bins score 0.9722
against 0.9535 for distant ones.

Good seeds are still weaker than trained embeddings. The CUDA path trains them
(`modules_to_save`, on by default in `build_peft_config`), so a GPU run should
learn better event representations than any MLX run can.

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

The tokenizer surgery itself is solved by `scripts/prepare_mlx_model.py`
(extend, seed, resize, convert) — under format v2 it registers 12,810 tokens.
Rebuild the local MLX model after any grammar change, or every event token
fragments into BPE pieces again.

## Base model access

`config.BASE_MODEL` is `Qwen/Qwen2.5-7B-Instruct` (ungated). The gated
`meta-llama/Meta-Llama-3.1-8B-Instruct` returns 403 on weight download from
this machine — its metadata reads fine, which is misleading. Ungated
alternatives:

- `Qwen/Qwen2.5-7B-Instruct` (the plan allows Llama *or* Qwen; the default)
- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` (MLX, 4-bit, stock tokenizer)

## Recommendation

| Goal | Route |
|---|---|
| Verify the pipeline runs | PyTorch/MPS, tiny model, `--limit-*` data |
| Iterate on an 8B locally | MLX + 4-bit, after fixing the tokenizer |
| The real 2M-example run | Rented CUDA GPU — bf16, as the plan intends |

### Why the full run does not belong here

The export is 2,018,334 examples. At ~200 tokens each with the extended
tokenizer that is ~400M tokens per epoch; at the measured 554 tokens/sec:

Measured at 1.88 examples/sec with the extended tokenizer:

| Scope | Extended tokenizer | Stock tokenizer |
|---|---|---|
| Full epoch (2,018,334 examples) | **12.4 days** | 29.6 days |
| 50k-example subset | **7.4 hours** | 17.6 hours |
| 20k-example subset | **3 hours** | 7 hours |

So the Mac is genuinely useful for a subset run overnight, and genuinely
unsuitable for the full corpus. Rent the GPU for the real thing.

### Running it

```bash
# one-off: build the base model with the grammar baked in
uv run python scripts/prepare_mlx_model.py --out models/qwen25-7b-typeshi-mlx

# mlx-lm wants a directory holding train.jsonl and valid.jsonl
uv run python -m mlx_lm lora \
  --model models/qwen25-7b-typeshi-mlx --train --data <data-dir> \
  --batch-size 1 --num-layers 8 --iters 5000 --max-seq-length 1024 \
  --adapter-path checkpoints/motor-mlx
```

Keep batch size at 1 — see the throughput table above.
