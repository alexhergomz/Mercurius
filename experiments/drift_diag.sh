#!/usr/bin/env bash
# Why is L1 (step 7875) worse than the 150-step R75 beyond 8k? (#66.2) Separate the
# INIT (sequential calibration on the training mix) from TRAINING DRIFT: RULER @32k and
# PG-19 to 64k on the long run's own snapshots. Step 0 = same init as L1, untrained.
set -uo pipefail
cd "$(dirname "$0")/.."
CV=cache/mla_seqcal_a860e70514_covs.pt; GR=cache/mla_seqcal_a860e70514_groups.json
for S in 0 1575 4725; do
  CK=ckpt/adapters-c0-long75-step$S.pt
  echo "[drift] step $S  $(date '+%a %H:%M:%S')"
  .venv/bin/python -m mercurius.eval.ruler --arms "S$S=$CK" --dial c0 --dc 512 --mem-cap-gb 40 \
    --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR \
    --lengths 32768 --samples 25 --em-samples 10 --out logs/ruler_drift_S$S.json > logs/ruler_drift_S$S.log 2>&1
  echo "[drift] ruler S$S exit $?"
  .venv/bin/python -m mercurius.eval.longppl --arms "S$S=$CK" --dial c0 --dc 512 --mem-cap-gb 40 \
    --fast-infer --quantize --merge-eval --covs $CV --mla-groups $GR --max-len 65536 --books 12 \
    --out logs/longppl64k_S$S.json --save-pos logs/longppl64k_S${S}_pos.npz > logs/longppl64k_S$S.log 2>&1
  echo "[drift] longppl S$S exit $?"
done
echo "[drift] ALL DONE  $(date '+%a %H:%M:%S')"
