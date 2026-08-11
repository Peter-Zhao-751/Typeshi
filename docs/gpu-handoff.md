# GPU Handoff — Phase-1 Motor Fine-Tune

Everything needed to move the Phase-1 training run onto a rented GPU. The
pipeline, dataset builder, and eval harness are done and tested; what remains
is the fine-tune itself, which does not fit on a Mac (see
`training-on-apple-silicon.md`).

## 1. What to rent

The run needs one GPU with **at least 48 GB**, and 80 GB is comfortable.
Multi-GPU buys nothing here — do not complicate the setup with it.

Memory budget for Qwen2.5-7B with the extended 164,874-token vocabulary
(152,064 base + 12,810 event/knob tokens, format v2):

| Component | Size |
|---|---|
| Base weights, bf16 | 15.4 GB |
| Gradients for trainable params, bf16 | 2.4 GB |
| AdamW moments, fp32 | 9.8 GB |
| fp32 master copy | 4.9 GB |
| **Subtotal** | **32.5 GB** + activations |

The ~1.2B trainable parameters are mostly `embed_tokens` and `lm_head`
(591M each), which are trained because the 12,810 event tokens are new — see
"Why the embeddings are trainable" below.

| Instance | Verdict |
|---|---|
| **1× H100 80GB** | Recommended — fastest, large margin |
| **1× A100 80GB** | Fine, usually cheaper |
| 1× A6000 48GB | Workable; keep batch small |
| 1× A100 40GB | Tight; needs `--freeze-embeddings` or grad checkpointing |
| 1× A10 24GB | Too small |

Check current Lambda pricing yourself — it moves. Budget on the order of a
day's rental for one epoch plus eval.

## 2. What to move

**Just the repository.** Do not upload the corpora or the processed dataset.

- The Aalto corpus is a single public URL and downloads in minutes.
- The dataset rebuild takes ~25 minutes, measured at 115 files/sec.
- Rebuilding also regenerates `data/processed/split.json`, which the eval
  requires. **That file cannot be reconstructed from the JSONL**, because the
  JSONL carries no writer IDs — so copying the processed data across and
  skipping the rebuild would leave the eval unable to identify held-out
  writers.

Transfer by pushing the branch and cloning, or directly:

```bash
rsync -avz --exclude data --exclude .venv --exclude models \
      --exclude checkpoints --exclude .git/objects/pack/tmp \
      ~/Desktop/Typeshi/ ubuntu@<host>:~/Typeshi/
```

KLiCKe is **not needed** for Phase 1: it supplies composition data (29,167 of
the 2,018,334 examples) and Phase 1 trains on transcription only. It has no
public direct URL, so leave it until Phase 2 and upload it then.

## 3. Setup

```bash
cd ~/Typeshi
bash scripts/setup_gpu.sh
```

That installs uv, builds a Python 3.12 virtualenv (**not 3.14 — torch ships no
stable wheels for it**), installs the CUDA torch stack, downloads and extracts
Aalto, runs the test suite, and rebuilds the dataset with its writer split.

## 4. Train

Smoke test before committing to a long run:

```bash
python -m typeshi.train_motor --mode transcription \
  --out checkpoints/smoke --epochs 0.002
```

Confirm the log shows `backend: {'dtype': 'bfloat16', ... 'bf16': True}`,
`seeded 12810 event-token embeddings`, and a loss that moves. Then:

```bash
python -m typeshi.train_motor --mode transcription --out checkpoints/motor \
  2>&1 | tee logs/train.log
```

**Consider a subset first.** The transcription split is 1,989,167 examples of
~108 tokens each (format v2) — around 215M tokens per epoch. Phase 1 only has
to learn motor timing, and that likely saturates well before a full pass.
Rebuild with `--limit-aalto 20000` for a ~230k-example run, check the eval,
and scale up only if the numbers demand it.

## 5. Evaluate

```bash
python scripts/run_eval.py --checkpoint checkpoints/motor --n 200 \
  --out eval_report.json
```

Speed tip: pointing `--held-out` at the full corpus works (non-held-out
writers are skipped) but parses every file it passes. Faster: symlink just
the held-out writers' files into a directory and point `--held-out` there —
the writer list is `test_writers` in the checkpoint's `split.json`.

Tier-1 passes only when ALL five gates hold (each closes a demonstrated
exploit — an adversarial review showed the original two-gate version passed
on exact-copy fakes and on cherry-picked 1% survivor generations):

- `pass_discriminator_has_teeth` — ≥ 0.90 vs the naive heuristic baseline
- `pass_serial_dependence_teeth` — ≥ 0.75 vs timing-shuffled REAL sessions;
  catches discriminators that only read marginal distributions
- `pass_model` — paired grouped-CV accuracy vs our generations in
  **[0.40, 0.55]**. The lower bound matters: unpaired CV on paired data
  scored 0.085 on exact copies, and below-chance means leakage, not realism
- `pass_generation_validity` — ≥ 90% of attempts parse AND type the target
  (replay similarity ≥ 0.8, transcription events only)
- `pass_control_near_chance` — real-vs-real in [0.40, 0.60]

`discriminator_accuracy_vs_model_timing_only` reports the same comparison
without event-count features; a large gap between it and the full number
means the model is being caught on length, not timing.

The eval refuses to run without a writer split — that is deliberate.
Scoring realism on writers the model trained on would pass Tier-1 for the
wrong reason. Training copies `split.json` into the checkpoint directory and
the eval prefers that bound copy, so a later dataset rebuild cannot swap the
held-out writers under an existing checkpoint.

## 6. Bring back

```bash
rsync -avz ubuntu@<host>:~/Typeshi/checkpoints/motor/ ./checkpoints/motor/
rsync -avz ubuntu@<host>:~/Typeshi/eval_report.json ./
```

The adapter is small; the resized embedding table is not, so expect a few GB.

---

## Why the embeddings are trainable

`build_peft_config()` sets `modules_to_save=["embed_tokens", "lm_head"]`. This
is deliberate and costs ~13 GB of the memory budget above.

The event grammar adds 12,810 tokens no base vocabulary has seen. Two problems
follow from resizing alone:

1. **LoRA never touches the embedding table.** With only `q/k/v/o_proj`
   adapted, every `<c:h>` and `<DT:k>` would keep its initial
   vector for the whole run, and `lm_head` could never learn to emit them —
   even though they are the entire output vocabulary.
2. **Resizing makes them indistinguishable.** `resize_token_embeddings` draws
   each new row from one fitted distribution; measured
   `cos(<DT:50>, <DT:120>) = 1.0000` before intervention.

`initialize_new_token_embeddings()` seeds each row from the sub-word pieces the
token used to split into, which restores the ordinal structure the time bins
depend on (adjacent bins 0.9722, distant 0.9535). Training then refines them.

If the GPU cannot hold it, `--freeze-embeddings` falls back to seeded-only
embeddings, which is what the MLX path is stuck with. Expect worse timing
fidelity, and say so in any writeup.

## Base model

`config.BASE_MODEL` is `Qwen/Qwen2.5-7B-Instruct` — ungated, no login needed,
and permitted by the plan ("Llama or Qwen"). It is the recommended GPU
default: plain attention, untied embeddings, and the whole pipeline is
tested against it.

`Qwen/Qwen3.5-9B` is a viable alternative (the local runs used its 0.8B
sibling; the LoRA target superset and tied-embedding handling cover the
hybrid architecture) — but on CUDA install `flash-linear-attention` and
`causal-conv1d` first, or its linear-attention layers fall back to a slow
torch path.

`meta-llama/Meta-Llama-3.1-8B-Instruct` is gated. Its metadata reads fine for
anyone, but downloading weights returns 403 without an approved access request,
so do not assume access from a successful `model_info` call. If approval comes
through, set `BASE_MODEL = LLAMA_BASE_MODEL` and `huggingface-cli login`.

## Licensing

Aalto is **non-commercial research use with attribution** (cite Dhakal, Feit,
Kristensson & Oulasvirta, CHI 2018, doi 10.1145/3173574.3174220). A checkpoint
trained on it must not be deployed commercially. KLiCKe ships no terms file at
all — resolve that before anything trained on it is published.
