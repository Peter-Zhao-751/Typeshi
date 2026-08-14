#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/Typeshi
source .venv/bin/activate
while pgrep -f converge_probe >/dev/null; do sleep 30; done
echo "==> [$(date +%T)] smoke: composition eval n=10"
python scripts/run_eval_composition.py --n 10 --out eval_report_composition_smoke.json \
  > logs/eval-comp-smoke.log 2>&1
if ! grep -q '"convergence_rate"' eval_report_composition_smoke.json 2>/dev/null; then
  echo "==> SMOKE FAILED; stopping"; exit 1
fi
echo "==> [$(date +%T)] full: composition eval n=100"
python scripts/run_eval_composition.py --n 100 --out eval_report_composition.json \
  > logs/eval-comp-full.log 2>&1
echo "==> [$(date +%T)] TIER2 CHAIN DONE"
