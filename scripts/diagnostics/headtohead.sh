#!/usr/bin/env bash
# Same 227 held-out writers, same n, both checkpoints: is motor-phase2's
# transcription realism as good as motor-full's, or did composition cost it?
set -uo pipefail
cd /home/ubuntu/Typeshi
source .venv/bin/activate
until grep -q "WINDOWED CHAIN DONE" logs/windowed-chain.log 2>/dev/null; do sleep 120; done
for ck in motor-full motor-phase2; do
  echo "==> [$(date +%T)] Tier-1 on shared writers: $ck"
  python scripts/run_eval.py --checkpoint "checkpoints/$ck" \
    --held-out data/heldout_shared --n 150 \
    --out "eval_report_shared_${ck}.json" > "logs/eval-shared-${ck}.log" 2>&1
done
echo "==> [$(date +%T)] HEAD TO HEAD DONE"
