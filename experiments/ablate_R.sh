#!/usr/bin/env bash
# #66.2: is the long-range loss CAUSED by the per-head query maps routing content into
# slow RoPE pairs? Swap ablations on the step-7875 model (everything else unchanged):
#   Rident    all R = I               Rslowprot  slow pairs (17-31, + their halves) identity, no mixing in/out
#   Rrotprot  all 64 rotary dims identity, no mixing in/out
set -uo pipefail
cd "$(dirname "$0")/.."
PID=${1:-0}; while [ "$PID" != 0 ] && kill -0 "$PID" 2>/dev/null; do sleep 60; done
CV=cache/mla_seqcal_a860e70514_covs.pt; GR=cache/mla_seqcal_a860e70514_groups.json
for V in Rident Rslowprot Rrotprot; do
  CK=ckpt/ablate/L1-$V.pt
  echo "[abl] $V  $(date '+%a %H:%M:%S')"
  .venv/bin/python -m mercurius.eval.longppl --arms "$V=$CK" --dial c0 --dc 512 --mem-cap-gb 40 \
    --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR --max-len 65536 --books 12 \
    --out logs/longppl64k_$V.json --save-pos logs/longppl64k_${V}_pos.npz > logs/longppl64k_$V.log 2>&1
  echo "[abl] longppl $V exit $?"
  .venv/bin/python -m mercurius.eval.ruler --arms "$V=$CK" --dial c0 --dc 512 --mem-cap-gb 40 \
    --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR \
    --lengths 32768 --samples 25 --em-samples 10 --out logs/ruler_abl_$V.json > logs/ruler_abl_$V.log 2>&1
  echo "[abl] ruler $V exit $?"
done
echo "[abl] ALL DONE  $(date '+%a %H:%M:%S')"
