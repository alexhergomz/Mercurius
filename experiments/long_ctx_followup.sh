#!/usr/bin/env bash
# After the leg boundary (#65): does the leg-1 model lose long-range ability BEYOND the 8k
# training length, relative to the 150-step 75% arm and the original? (user, 2026-10-02:
# the 16k RULER value-NLL gap is small -- measure at lengths long enough to see effects.)
#   1. position-bucketed NLL on 12 PG-19 test books (public domain) to 128k, per-position
#      NLL saved -> paired per-book / per-block comparison
#   2. RULER 32k / 64k, 25 samples (EM 10), same tasks as the 4-16k runs
# Converted arms run with VeRA MERGED into bf16 (--merge-eval, ~2.5x faster; replay of L1
# merged: CE 1.9613 / 2.1952 vs trained 1.9612 / 2.1953). --fast-infer (models/fast_infer.py):
# gold span scored in ONE chunked forward continuing the cache, greedy EM stops once every
# gold is present (EM unchanged by construction); experiments/check_fast_infer.py: dNLL
# <= 0.004, EM agree 18/18, ~25-35% faster before the early stop.
# Arms: L1 = c0-long75 step 7875; R75 = the matched 150-step plain-75% arm; ORIG = NF4
# original (the matching baseline). Each converted arm with ITS OWN covs / groups.
#   nohup bash experiments/long_ctx_followup.sh <boundary pid> > logs/long_ctx_followup.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PID=${1:?boundary script pid}
say() { echo "[long] $*  $(date '+%a %H:%M:%S')"; }
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
say "boundary finished; starting long-context evals"
L1="L1 ckpt/adapters-c0-long75-step7875.pt cache/mla_seqcal_a860e70514_covs.pt cache/mla_seqcal_a860e70514_groups.json"
R75="R75 ckpt/adapters-c0-mol1-care-150-c75.pt cache/kv_covs_4b_mix.pt cache/mla_groups_ungrouped_4096.json"
ORIG="ORIG ORIGINAL_NF4 - -"
for row in "$L1" "$R75" "$ORIG"; do
  set -- $row
  ARGS=(--dial c0 --dc 512 --mem-cap-gb 40 --fast-infer)
  [ "$2" = ORIGINAL_NF4 ] || ARGS+=(--quantize --merge-eval --covs "$3" --mla-groups "$4")
  say "longppl $1"
  .venv/bin/python -m mercurius.eval.longppl --arms "$1=$2" "${ARGS[@]}" \
    --max-len 131072 --books 12 --out "logs/longppl128k_$1.json" \
    --save-pos "logs/longppl128k_$1_pos.npz" > "logs/longppl128k_$1.log" 2>&1
  say "longppl $1 exit $?"
  grep -E "OOM" "logs/longppl128k_$1.log" | sed 's/^/[long] /'
done
for row in "$L1" "$R75" "$ORIG"; do
  set -- $row
  ARGS=(--dial c0 --dc 512 --mem-cap-gb 40 --fast-infer)
  [ "$2" = ORIGINAL_NF4 ] || ARGS+=(--quantize --merge-eval --covs "$3" --mla-groups "$4")
  say "RULER 32k/64k $1"
  .venv/bin/python -m mercurius.eval.ruler --arms "$1=$2" "${ARGS[@]}" \
    --lengths 32768 65536 --samples 25 --em-samples 10 \
    --out "logs/ruler_long_$1.json" > "logs/ruler_long_$1.log" 2>&1
  say "RULER $1 exit $?"
done
say "ALL DONE"
