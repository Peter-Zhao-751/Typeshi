#!/usr/bin/env bash
# Re-score both checkpoints under the corrected protocol: writer-grouped CV
# folds and at most 3 sessions per participant.
set -uo pipefail
cd /home/ubuntu/Typeshi
source .venv/bin/activate
until ! pgrep -f run_eval_composition >/dev/null; do sleep 60; done
for ck in motor-full motor-phase2; do
  echo "==> [$(date +%T)] corrected-protocol Tier-1: $ck"
  python scripts/run_eval.py --checkpoint "checkpoints/$ck" \
    --held-out data/heldout_writers --n 200 --max-per-writer 3 \
    --out "eval_report_${ck}_writergrouped.json" \
    > "logs/eval-${ck}-writergrouped.log" 2>&1
done
echo "==> [$(date +%T)] CORRECTED TIER1 DONE"
