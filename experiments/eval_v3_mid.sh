#!/usr/bin/env bash
# #68.5: did the RoPE fix + absorbable MLA fix c0-long75's long-range retrieval loss?
# Pre-QAT snapshots of v3-absorb, same settings as the L1 / R75 / ORIG long-range evals.
set -uo pipefail
cd "$(dirname "$0")/.."
CV=cache/mla_seqcal_667126e299_covs.pt; GR=cache/mla_seqcal_667126e299_groups.json
say() { echo "[ev] $*  $(date '+%a %H:%M:%S')"; }
say "replay check step 1575"
.venv/bin/python experiments/replay_check.py --arm "V1575=ckpt/adapters-v3-absorb-step1575.pt" \
  --groups $GR --covs $CV --dial c0 --expect 1.9583 2.1833 --merge-eval > logs/replay_v3_1575.log 2>&1
RC=$?; grep "\[replay\]" logs/replay_v3_1575.log | sed 's/^/[ev] /'
[ $RC -eq 0 ] || { say "REPLAY MISMATCH -- not evaluating"; tail -5 logs/replay_v3_1575.log; exit 1; }
for S in 3150 1575; do
  CK=ckpt/adapters-v3-absorb-step$S.pt
  say "RULER 32k V$S"
  .venv/bin/python -m mercurius.eval.ruler --arms "V$S=$CK" --dial c0 --dc 512 --mem-cap-gb 30 \
    --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR \
    --lengths 32768 --samples 25 --em-samples 10 --out logs/ruler_v3_$S.json > logs/ruler_v3_$S.log 2>&1
  say "RULER V$S exit $?"
  say "longppl 64k V$S"
  .venv/bin/python -m mercurius.eval.longppl --arms "V$S=$CK" --dial c0 --dc 512 --mem-cap-gb 30 \
    --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR --max-len 65536 --books 12 \
    --out logs/longppl64k_V$S.json --save-pos logs/longppl64k_V${S}_pos.npz > logs/longppl64k_V$S.log 2>&1
  say "longppl V$S exit $?"
done
say "ALL DONE"
