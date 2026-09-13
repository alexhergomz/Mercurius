#!/usr/bin/env bash
# Long-context eval, then Stage D once it completes.
set -u
cd /home/srdelam/qwen-surgery
source env.sh

echo "=== [1/2] long-context NIAH ==="
python -u src/eval_longctx.py \
  --lengths 32768 65536 131072 \
  --depths 0.1 0.5 0.9 \
  > logs/longctx.log 2>&1
echo "long-context eval exit: $?"

if [ ! -f src/run_stage_d.py ]; then
  echo "stage D runner missing, stopping"; exit 1
fi
./src/diskguard.sh || { echo "disk guard tripped, not starting stage D"; exit 1; }

echo "=== [2/2] stage D: TransMLA latent KV ==="
python -u src/run_stage_d.py > logs/stage_d.log 2>&1
echo "stage D exit: $?"
