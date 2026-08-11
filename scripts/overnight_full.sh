#!/usr/bin/env bash
# Full-corpus epoch + codec-fair eval, chained so the overnight run scores
# itself. Launch detached: setsid nohup bash scripts/overnight_full.sh &
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "==> [$(date +%F' '%T)] training: full corpus, 1 epoch"
python -m typeshi.train_motor --mode transcription \
  --data data/processed_full/train.jsonl --out checkpoints/motor-full \
  --batch 32 --accum 1 2>&1 | tee logs/train-gpu-full.log
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
  echo "==> [$(date +%F' '%T)] TRAINING FAILED ($status); no eval" >&2
  exit "$status"
fi

echo "==> [$(date +%F' '%T)] eval: 200 held-out sessions"
python scripts/run_eval.py --checkpoint checkpoints/motor-full \
  --held-out data/heldout_writers --n 200 --out eval_report_full.json \
  2>&1 | tee logs/eval-full.log
echo "==> [$(date +%F' '%T)] OVERNIGHT CHAIN DONE"
