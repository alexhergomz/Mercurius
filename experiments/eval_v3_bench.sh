#!/usr/bin/env bash
# #68.8 GSM8K / HumanEval / MBPP on the final deployed 4-bit v3 (greedy, no-think, 768 max-new).
set -uo pipefail
cd "$(dirname "$0")/.."
CV=cache/mla_seqcal_667126e299_covs.pt; GR=cache/mla_seqcal_667126e299_groups.json
CK=ckpt/adapters-v3-absorb-step3900.pt
Q="--qat --qat-kv-bits 4 --qat-kv-group 32 --qat-kv-quant tq --qat-kv-rot none --qat-gate-bits 16 --qat-embed-bits 4"
say() { echo "[bench] $*  $(date '+%a %H:%M:%S')"; }
for row in "humaneval mbpp:16" "gsm8k:32"; do
  T=${row%%:*}; B=${row#*:}
  say "gen [$T] batch $B"
  .venv/bin/python experiments/bench_full.py --tasks $T --limit 0 --max-new 768 --temperature 0 \
    --no-think --dc 512 --covs $CV --mla-groups $GR --dial c0 $Q --batch $B \
    --arms "V3900=$CK" --out logs/gen_greedy_v3.json > "logs/gen_V3900_${T// /-}.log" 2>&1
  say "gen [$T] exit $?"; grep -E "^  V3900 [a-z0-9]+: " "logs/gen_V3900_${T// /-}.log" | sed 's/^/[bench] /'
done
say "ALL DONE"
