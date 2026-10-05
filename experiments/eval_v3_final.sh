#!/usr/bin/env bash
# #68.7 final evaluation of v3-absorb step 3900 AS DEPLOYED (4-bit QAT form: NF4 weights,
# TurboQuant-4 latent + RoPE key with the saved rotations, NF4 embedding, bf16 gates).
set -uo pipefail
cd "$(dirname "$0")/.."
CV=cache/mla_seqcal_667126e299_covs.pt; GR=cache/mla_seqcal_667126e299_groups.json
CK=ckpt/adapters-v3-absorb-step3900.pt
Q="--qat --qat-kv-bits 4 --qat-kv-group 32 --qat-kv-quant tq --qat-kv-rot none --qat-gate-bits 16 --qat-embed-bits 4"
say() { echo "[fin] $*  $(date '+%a %H:%M:%S')"; }
say "replay (deployed 4-bit)"
.venv/bin/python experiments/replay_check.py --arm "V3900=$CK" --groups $GR --covs $CV --dial c0 \
  --expect 1.9743 2.2089 $Q > logs/replay_v3_3900.log 2>&1
RC=$?; grep "\[replay\]" logs/replay_v3_3900.log | sed 's/^/[fin] /'
[ $RC -eq 0 ] || { say "REPLAY MISMATCH -- stop"; tail -8 logs/replay_v3_3900.log; exit 1; }
COMMON="--dial c0 --dc 512 --mem-cap-gb 60 --fast-infer --quantize --covs $CV --mla-groups $GR $Q"
say "RULER 32k"
.venv/bin/python -m mercurius.eval.ruler --arms "V3900=$CK" $COMMON --lengths 32768 \
  --samples 25 --em-samples 10 --out logs/ruler_v3_3900_32k.json > logs/ruler_v3_3900_32k.log 2>&1
say "RULER 32k exit $?"
say "longppl 64k"
.venv/bin/python -m mercurius.eval.longppl --arms "V3900=$CK" $COMMON --max-len 65536 --books 12 \
  --out logs/longppl64k_V3900.json --save-pos logs/longppl64k_V3900_pos.npz > logs/longppl64k_V3900.log 2>&1
say "longppl exit $?"
say "RULER 64k"
.venv/bin/python -m mercurius.eval.ruler --arms "V3900=$CK" $COMMON --lengths 65536 \
  --samples 25 --em-samples 10 --out logs/ruler_v3_3900_64k.json > logs/ruler_v3_3900_64k.log 2>&1
say "RULER 64k exit $?"
for T in "humaneval mbpp" "gsm8k"; do
  say "gen [$T]"
  .venv/bin/python experiments/bench_full.py --tasks $T --limit 0 --max-new 768 --temperature 0 \
    --no-think --dc 512 --covs $CV --mla-groups $GR --dial c0 $Q \
    --arms "V3900=$CK" --out logs/gen_greedy_v3.json > "logs/gen_V3900_${T// /-}.log" 2>&1
  say "gen [$T] exit $?"; grep -E "^  V3900 [a-z0-9]+: " "logs/gen_V3900_${T// /-}.log" | sed 's/^/[fin] /'
done
say "ALL DONE"
