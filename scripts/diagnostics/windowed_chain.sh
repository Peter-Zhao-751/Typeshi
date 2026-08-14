#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/Typeshi
source .venv/bin/activate
until grep -q "NIGHT CHAIN DONE\|FAILED" logs/night-chain.log 2>/dev/null; do sleep 120; done
echo "==> [$(date +%T)] windowed Tier-2 eval n=100"
python scripts/run_eval_composition.py --n 100 --out eval_report_composition_windowed.json \
  > logs/eval-comp-windowed.log 2>&1
echo "==> [$(date +%T)] WINDOWED CHAIN DONE"
