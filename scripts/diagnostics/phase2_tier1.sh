#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/Typeshi
source .venv/bin/activate
until grep -q "WINDOWED CHAIN DONE" logs/windowed-chain.log 2>/dev/null; do sleep 120; done
echo "==> [$(date +%T)] Tier-1 transcription eval of motor-phase2"
python scripts/run_eval.py --checkpoint checkpoints/motor-phase2 \
  --held-out data/heldout_writers --n 200 --out eval_report_phase2_tier1.json \
  > logs/eval-phase2-tier1.log 2>&1
echo "==> [$(date +%T)] PHASE2 TIER1 DONE"
