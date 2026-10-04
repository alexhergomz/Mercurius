#!/usr/bin/env bash
# After the long-context RULER (#65): (1) QAT memory/throughput probes from COPIES of the
# leg-1 resume file -- 8k-only and 32k-only, 3 steps each, logged every step -- because the
# boundary's QAT smoke OOMed in backward under the 24 GiB cap; (2) longppl for the two
# students at 64k (128k OOMed for them under the 40 GiB eval cap; positions <= 64k of the
# ORIG 128k run are directly comparable: NLL at t depends only on the prefix).
#   nohup bash experiments/after_followup.sh <followup pid> > logs/after_followup.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PID=${1:?followup pid}
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
. logs/leg2_choice.env
# TurboQuant-MSE KV (#66, user): PTQ cost on the leg-1 model vs the int4 g32 reference
.venv/bin/python experiments/qat_ptq_sweep.py --adapters ckpt/adapters-c0-long75-step7875.pt \
  --covs cache/mla_seqcal_a860e70514_covs.pt --groups cache/mla_seqcal_a860e70514_groups.json \
  --only-tq --out logs/qat_ptq_tq.json > logs/qat_ptq_tq.log 2>&1
echo "[after] TQ PTQ exit $?"; grep "^\[ptq\]" logs/qat_ptq_tq.log | sed 's/^/[after] /'
for L in 8192 32768; do
  cp -p ckpt/resume-c0-long75.pt ckpt/resume-c0-long75-probe$L.pt
  echo "[after] probe $L  $(date '+%a %H:%M:%S')"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TAG=c0-long75-probe$L STOPAT=7878 \
    LENMIX=$L:1.0 LOGEVERY=1 MEMCAP=${MEMCAP:-44} KVROT=$KVROT GATEBITS=$GATEBITS \
    bash experiments/run_long_75.sh --go --resume > logs/run-c0-long75-probe$L.log 2>&1
  echo "[after] probe $L exit $?"
  grep -E "length mix|step +787[6-8]/|STOPPED|OutOfMemory|refusing" logs/run-c0-long75-probe$L.log | sed 's/^/[after] /'
  rm -f ckpt/resume-c0-long75-probe$L.pt ckpt/resume-c0-long75-probe$L-leg1end.pt
done
L1="L1 ckpt/adapters-c0-long75-step7875.pt cache/mla_seqcal_a860e70514_covs.pt cache/mla_seqcal_a860e70514_groups.json"
R75="R75 ckpt/adapters-c0-mol1-care-150-c75.pt cache/kv_covs_4b_mix.pt cache/mla_groups_ungrouped_4096.json"
for row in "$L1" "$R75"; do
  set -- $row
  echo "[after] longppl64k $1  $(date '+%a %H:%M:%S')"
  .venv/bin/python -m mercurius.eval.longppl --arms "$1=$2" --dial c0 --dc 512 --mem-cap-gb 40 \
    --fast-infer --quantize --merge-eval --covs "$3" --mla-groups "$4" \
    --max-len 65536 --books 12 --out "logs/longppl64k_$1.json" \
    --save-pos "logs/longppl64k_$1_pos.npz" > "logs/longppl64k_$1.log" 2>&1
  echo "[after] longppl64k $1 exit $?"; grep -c OOM "logs/longppl64k_$1.log" | sed 's/^/[after] OOM books: /'
done
echo "[after] ALL DONE  $(date '+%a %H:%M:%S')"
