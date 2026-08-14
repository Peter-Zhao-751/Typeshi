#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/Typeshi
source .venv/bin/activate
until grep -q "TIER2 CHAIN DONE\|SMOKE FAILED" logs/tier2-chain.log 2>/dev/null; do sleep 60; done
echo "==> [$(date +%T)] RAFT smoke (10 targets, k=2)"
python scripts/gen_raft_data.py --targets 10 --k 2 --out data/raft_smoke/train.jsonl \
  > logs/raft-smoke.log 2>&1
if ! grep -q "RAFT DATA DONE" logs/raft-smoke.log || [ ! -s data/raft_smoke/train.jsonl ]; then
  echo "==> RAFT SMOKE FAILED; night chain stops"; exit 1
fi
echo "==> [$(date +%T)] RAFT full (800 targets, k=4)"
python scripts/gen_raft_data.py --targets 800 --k 4 --out data/raft/train.jsonl \
  > logs/raft-full.log 2>&1
grep -q "RAFT DATA DONE" logs/raft-full.log || { echo "==> RAFT FULL FAILED"; exit 1; }
cp data/processed_full/split.json data/raft/split.json
echo "==> [$(date +%T)] RAFT SFT from motor-full"
python -m typeshi.train_motor --mode transcription --data data/raft/train.jsonl \
  --init-adapter checkpoints/motor-full --out checkpoints/motor-raft \
  --epochs 2 --batch 32 --accum 1 > logs/train-raft.log 2>&1
grep -q "train_runtime" logs/train-raft.log || { echo "==> RAFT SFT FAILED"; exit 1; }
echo "==> [$(date +%T)] Tier-1 re-eval of motor-raft"
python scripts/run_eval.py --checkpoint checkpoints/motor-raft \
  --held-out data/heldout_writers --n 200 --out eval_report_raft.json \
  > logs/eval-raft.log 2>&1
echo "==> [$(date +%T)] NIGHT CHAIN DONE"
