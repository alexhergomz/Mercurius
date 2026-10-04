#!/usr/bin/env bash
# #66.3: evaluate the no-per-head-q control at step 1575 exactly like the L1 snapshots.
set -uo pipefail
cd "$(dirname "$0")/.."
PID=${1:?trainer pid}
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
grep -q "STOPPED at step 1575" logs/run-c0-long75-noR.log || { echo "[noR] did not stop cleanly"; tail -5 logs/run-c0-long75-noR.log; exit 1; }
CV=cache/mla_seqcal_a860e70514_covs.pt; GR=cache/mla_seqcal_a860e70514_groups.json
CK=ckpt/adapters-c0-long75-noR-step1575.pt
.venv/bin/python -m mercurius.eval.ruler --arms "N1575=$CK" --dial c0 --dc 512 --mem-cap-gb 40 \
  --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR \
  --lengths 32768 --samples 25 --em-samples 10 --out logs/ruler_drift_N1575.json > logs/ruler_drift_N1575.log 2>&1
echo "[noR] ruler exit $?"
.venv/bin/python -m mercurius.eval.longppl --arms "N1575=$CK" --dial c0 --dc 512 --mem-cap-gb 40 \
  --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR --max-len 65536 --books 12 \
  --out logs/longppl64k_N1575.json --save-pos logs/longppl64k_N1575_pos.npz > logs/longppl64k_N1575.log 2>&1
echo "[noR] longppl exit $?  $(date '+%a %H:%M:%S')"
