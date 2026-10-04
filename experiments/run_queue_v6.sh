#!/usr/bin/env bash
# Model line 1 long run, then GSM8K at four budgets -> the token-budget curve.
# Waits out queue_v4 (step0-rope8094) first; both poll for a free GPU and would
# otherwise contend.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "run_queue_v4[.]sh" >/dev/null; do sleep 60; done
while pgrep -f "mercurius.recovery.train|bench_full[.]py" >/dev/null; do sleep 60; done

if [ ! -f ckpt/adapters-line1-rope6000.pt ]; then
  echo "[q6] === training line1-rope6000 (6000 steps, ~30 h)  $(date '+%F %H:%M:%S')"
  bash experiments/run_line1_rope6000.sh > logs/run-line1-rope6000.log 2>&1
fi
if [ ! -f ckpt/adapters-line1-rope6000.pt ]; then
  echo "[q6] FAILED training -- see logs/run-line1-rope6000.log"; exit 1
fi

# GSM8K at four budgets. 6000 steps x ~6000 tok/step = 3 M / 9 M / 18 M / 36 M tokens.
for S in 500 1500 3000 6000; do
  if [ "$S" = 6000 ]; then CK=ckpt/adapters-line1-rope6000.pt; else CK="ckpt/adapters-line1-rope6000-step$S.pt"; fi
  [ -f "$CK" ] || { echo "[q6] skip step$S: $CK missing"; continue; }
  echo "[q6] === GSM8K line1-step$S  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt \
    --mla-groups cache/mla_groups_retr_4096_mix.json --dial c0 \
    --arms "line1-step$S=$CK" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "line1-step$S gsm8k:" logs/bench_gsm8k.log | tail -1
done
echo "[q6] done  $(date '+%F %H:%M:%S')"
