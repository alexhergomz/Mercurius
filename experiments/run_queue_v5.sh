#!/usr/bin/env bash
# Step-0 RoPE dial sweep, ordered cheapest-informative first. NO TRAINING in any arm.
# Fills in the curve between the two endpoints we already have:
#     keep  0 (nope) -> 75.4%      keep 32 (c0) -> 86.4%
# k4 first because MHA2MLA's default is 4 rotary subspaces and they report 4->8 as
# negligible; if k4 recovers most of the 11 points the fix is nearly free.
set -uo pipefail
cd "$(dirname "$0")/.."
# queue_v4 (step0-rope8094) also polls for a free GPU, so wait it out FIRST or
# both fire at once when the current bench ends and contend for the device.
while pgrep -f "run_queue_v4[.]sh" >/dev/null; do sleep 60; done
for DIAL in k4 k8 c1; do
  CK="ckpt/adapters-step0-dial-$DIAL.pt"
  if [ ! -f "$CK" ]; then
    while pgrep -f "mercurius.recovery.train|bench_full.py" >/dev/null; do sleep 60; done
    echo "[q5] === building step0-dial-$DIAL (no training)  $(date '+%H:%M:%S')"
    bash experiments/run_step0_dial.sh "$DIAL" > "logs/run-step0-dial-$DIAL.log" 2>&1
  fi
  [ -f "$CK" ] || { echo "[q5] FAILED build $DIAL -- see logs/run-step0-dial-$DIAL.log"; continue; }
  while pgrep -f "bench_full.py" >/dev/null; do sleep 60; done
  echo "[q5] === GSM8K step0-dial-$DIAL  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt \
    --mla-groups cache/mla_groups_retr_4096_mix.json --dial "$DIAL" \
    --arms "step0-dial-$DIAL=$CK" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "step0-dial-$DIAL gsm8k:" logs/bench_gsm8k.log | tail -1
done
echo "[q5] done  $(date '+%H:%M:%S')"
