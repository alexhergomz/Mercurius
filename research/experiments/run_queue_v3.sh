#!/usr/bin/env bash
# The one arm that was genuinely lost. ungrouped150 was killed at step 20/150 on
# 2026-09-26 21:29 when queue_v2 took the GPU; only its step-0 -best.pt survived.
# Every other 150-step arm (base150, mla8094, rope150, taps150, gate150) finished.
#
# WHAT IT ANSWERS. Same total rank and the same PER-LAYER rank as base150, but one
# group per layer instead of the water-filled partition -- so it isolates the
# BLOCK-DIAGONAL RESTRICTION from the rank allocation, which uniform150 conflated.
# Measured on base150 the off-block weights sit at 0.3-0.5% of on-block magnitude
# after 150 steps, using 5% of their reachable displacement, so the restriction
# does not lift itself. Grouping's justification was allocation, and allocation is
# now measured as worth ~0.06 nats and washing out -- so this asks whether grouping
# pays a permanent expressiveness cost for a benefit adaptation erases.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "[q3] === training ablate-ungrouped150  $(date '+%H:%M:%S')"
bash experiments/run_ablate_ungrouped150.sh > logs/run-ablate-ungrouped150.log 2>&1
if [ ! -f ckpt/adapters-ablate-ungrouped150.pt ]; then
  echo "[q3] FAILED training -- see logs/run-ablate-ungrouped150.log"; exit 1
fi
echo "[q3] === GSM8K ungrouped150  $(date '+%H:%M:%S')"
.venv/bin/python experiments/bench_full.py \
  --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
  --dc 512 --covs cache/kv_covs_4b_mix.pt \
  --mla-groups cache/mla_groups_ungrouped_4096.json --dial nope \
  --arms "ungrouped150=ckpt/adapters-ablate-ungrouped150.pt" \
  --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
grep -E "ungrouped150 gsm8k:" logs/bench_gsm8k.log | tail -1
echo "[q3] done  $(date '+%H:%M:%S')"
