# GPU Handoff — Phase-1 Motor Fine-Tune

Everything needed to move the Phase-1 training run onto a rented GPU. The
pipeline, dataset builder, and eval harness are done and tested; what remains
is the fine-tune itself, which does not fit on a Mac (see
`training-on-apple-silicon.md`).

## 1. What to rent

The run needs one GPU with **at least 48 GB**, and 80 GB is comfortable.
Multi-GPU buys nothing here — do not complicate the setup with it.

Memory budget for Qwen2.5-7B with the extended 152,019-token vocabulary:

| Component | Size |
|---|---|
| Base weights, bf16 | 15.2 GB |
| Gradients for trainable params, bf16 | 2.3 GB |
| AdamW moments, fp32 | 9.0 GB |
| fp32 master copy | 4.5 GB |
| **Subtotal** | **31.0 GB** + activations |

The 1.13B trainable parameters are mostly `embed_tokens` and `lm_head`
(545M each), which are trained because the 356 event tokens are new — see
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
`seeded 354 event-token embeddings`, and a loss that moves. Then:

```bash
python -m typeshi.train_motor --mode transcription --out checkpoints/motor \
  2>&1 | tee logs/train.log
```

**Consider a subset first.** The transcription split is 1,989,167 examples of
roughly 190–240 tokens each — around 400M tokens per epoch. Phase 1 only has
to learn motor timing, and that likely saturates well before a full pass.
Rebuild with `--limit-aalto 20000` for a ~230k-example run, check the eval,
and scale up only if the numbers demand it.

## 5. Evaluate

```bash
python scripts/run_eval.py --checkpoint checkpoints/motor --n 200 \
  --out eval_report.json
```

Tier-1 passes when both hold:

- `pass_discriminator_has_teeth` — accuracy ≥ 0.90 against the naive
  heuristic baseline, proving the discriminator can detect fake timing at all
- `pass_model` — accuracy ≤ 0.55 against our generations

Also read `discriminator_accuracy_real_vs_real_control`. It should sit near
chance; if it is high, the writer population itself is separable and the model
number is inflated rather than good.

The eval refuses to run without `data/processed/split.json` — that is
deliberate. Scoring realism on writers the model trained on would pass Tier-1
for the wrong reason.

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

The event grammar adds 356 tokens no base vocabulary has seen. Two problems
follow from resizing alone:

1. **LoRA never touches the embedding table.** With only `q/k/v/o_proj`
   adapted, every `<DT:k>`, `<KEY:c>`, and `<HOLD:k>` would keep its initial
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
and permitted by the plan ("Llama or Qwen").

`meta-llama/Meta-Llama-3.1-8B-Instruct` is gated. Its metadata reads fine for
anyone, but downloading weights returns 403 without an approved access request,
so do not assume access from a successful `model_info` call. If approval comes
through, set `BASE_MODEL = LLAMA_BASE_MODEL` and `huggingface-cli login`.

## Licensing

Aalto is **non-commercial research use with attribution** (cite Dhakal, Feit,
Kristensson & Oulasvirta, CHI 2018, doi 10.1145/3173574.3174220). A checkpoint
trained on it must not be deployed commercially. KLiCKe ships no terms file at
all — resolve that before anything trained on it is published.
