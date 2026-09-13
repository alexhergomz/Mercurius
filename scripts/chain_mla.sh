#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
source env.sh
if [ -n "123139" ]; then
  while kill -0 123139 2>/dev/null; do sleep 60; done
fi
echo "CARE finished; restarting MLA recovery with the cached top-K path"
./src/diskguard.sh || { echo "disk guard tripped"; exit 1; }
python -u src/train_recovery.py \
  --dial nope --seed-decay --seed-alpha 0 \
  --init-adapters ckpt/adapters-combined.pt \
  --mla-energy 0.99 --logit-cache cache/tk64_1m.pt --topk 64 \
  --length-mix --grad-checkpoint \
  --steps 400 --eval-every 100 --lr 2e-4 --tag mla-recover \
  > logs/train_mla.log 2>&1
echo "MLA recovery exit: $?"
