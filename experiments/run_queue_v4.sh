#!/usr/bin/env bash
# The missing cell of the 2x2, at step 0 so NO training is involved.
#
# Measured at step 0 (GDN-2 lift is exact at init, so it contributes zero here):
#             r=4096   r=8094
#   RoPE       86.4       ???     <- this run
#   NoPE       75.4      83.3
#
# Doubling rank buys +7.9 under NoPE, but compression costs only 4.1 IN TOTAL under
# RoPE -- so rank is worth at least 1.9x more when RoPE is gone. Additivity is
# already impossible (it predicts 94.3%, above the 90.5% unmodified base), so the
# effects are sub-additive; this cell says by how much, and whether rank is worth
# spending on at all once RoPE is kept.
#
# Costs one build (~15 min, no optimizer step) plus one GSM8K pass.
set -uo pipefail
cd "$(dirname "$0")/.."
# wait out whatever holds the GPU, without a timer: poll for the processes only
while pgrep -f "mercurius.recovery.train|bench_full.py" >/dev/null; do sleep 60; done
echo "[q4] === building step0-rope8094 (no training)  $(date '+%H:%M:%S')"
bash experiments/run_step0_rope8094.sh > logs/run-step0-rope8094.log 2>&1
if [ ! -f ckpt/adapters-step0-rope8094.pt ]; then
  echo "[q4] FAILED build -- see logs/run-step0-rope8094.log"; exit 1
fi
echo "[q4] === GSM8K step0-rope8094  $(date '+%H:%M:%S')"
.venv/bin/python experiments/bench_full.py \
  --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
  --dc 512 --covs cache/kv_covs_4b_mix.pt \
  --mla-groups cache/mla_groups_retr_8192_mix.json --dial c0 \
  --arms "step0-rope8094=ckpt/adapters-step0-rope8094.pt" \
  --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
grep -E "step0-rope8094 gsm8k:" logs/bench_gsm8k.log | tail -1
echo "[q4] done  $(date '+%H:%M:%S')"
