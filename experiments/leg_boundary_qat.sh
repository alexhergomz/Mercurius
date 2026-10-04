#!/usr/bin/env bash
# Leg boundary of c0-long75 (#64, #65): leg 1 stops at 7875 ->
#   1. replay check of the step-7875 adapters (the benchmarks must measure the trained model)
#   2. mid-run BENCHMARKS on step 7875, same settings as the 150-step arms (P75 / R75-plain):
#      RULER 4/8/16k (paired per-sample vs R75-plain and ORIG), greedy GSM8K, HumanEval, MBPP
#   3. PTQ cost of the deployed 4-bit format + KV rotation choice
#   4. 25-step QAT smoke on a COPY of the resume file (leg 2 readiness + throughput)
# and then STOPS: leg 2 starts only after the user has seen the benchmarks (user,
# 2026-10-01). Any failure stops here with the reason.
#   nohup bash experiments/leg_boundary_qat.sh <leg1 pid> > logs/leg_boundary.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PID=${1:?leg-1 trainer pid}
say() { echo "[boundary] $*  $(date '+%a %H:%M:%S')"; }
CV=cache/mla_seqcal_a860e70514_covs.pt
GR=cache/mla_seqcal_a860e70514_groups.json
CK=ckpt/adapters-c0-long75-step7875.pt
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
grep -q "STOPPED at step 7875" logs/run-c0-long75.log || { say "leg 1 did NOT stop cleanly at 7875 -- not continuing"; tail -5 logs/run-c0-long75.log; exit 1; }
[ -f "$CK" ] || { say "no step-7875 adapters"; exit 1; }
say "leg 1 stopped cleanly at 7875"

# 1. replay: rebuild exactly as the harnesses do, must match the training eval @7875
ce () { grep -A3 "\[eval @ 7875\]" logs/run-c0-long75.log | grep "@$1 " | head -1 | sed -E 's/.*CE +([0-9.]+).*/\1/'; }
E2=$(ce 2048); E8=$(ce 8192)
.venv/bin/python experiments/replay_check.py --arm "L1=$CK" --groups $GR --covs $CV \
  --dial c0 --expect $E2 $E8 > logs/replay_L1.log 2>&1
RC=$?; grep "\[replay\]" logs/replay_L1.log | sed 's/^/[boundary] /'
[ $RC -eq 0 ] || { say "REPLAY MISMATCH or error -- benchmarks would not measure the trained model; stopping"; tail -5 logs/replay_L1.log; exit 1; }

# 2. benchmarks (settings identical to queue14 RULER / queue16 generation)
say "RULER"
.venv/bin/python -m mercurius.eval.ruler --arms "L1=$CK" --mla-groups $GR \
  --quantize --dial c0 --covs $CV --dc 512 \
  --lengths 4096 8192 16384 --samples 50 --em-samples 20 \
  --out logs/ruler_long75_L1.json > logs/ruler_long75_L1.log 2>&1
say "RULER exit $?"
for T in "gsm8k" "humaneval mbpp"; do
  say "gen [$T]"
  .venv/bin/python experiments/bench_full.py --tasks $T --limit 0 --max-new 768 \
    --temperature 0 --no-think --dc 512 --covs $CV --mla-groups $GR --dial c0 \
    --arms "L1=$CK" --out logs/gen_greedy_long75.json > "logs/gen_L1_${T// /-}.log" 2>&1
  say "gen [$T] exit $?"
  grep -E "^  L1 [a-z0-9]+: " "logs/gen_L1_${T// /-}.log" | sed 's/^/[boundary] /'
done

# 3. PTQ decomposition + KV rotation choice
.venv/bin/python experiments/qat_ptq_sweep.py --adapters $CK --covs $CV --groups $GR \
  > logs/qat_ptq_sweep.log 2>&1 || { say "PTQ sweep failed"; tail -20 logs/qat_ptq_sweep.log; exit 1; }
grep "^\[ptq\]" logs/qat_ptq_sweep.log
KVROT=$(.venv/bin/python -c "
import json; r=json.load(open('logs/qat_ptq_sweep.json'))
print(min(('none','orth'), key=lambda o: r[f'+KV int4 g32 rot={o}']['8192']))")
# Gates deploy in their EXACT factored form (#64.1): tiled base (32 unique rows) + VeRA
# vectors + the shared bf16 A/B (+ the decay's rank-32 LoRA) -- 0.06 GB, lossless, and
# fewer bytes/token than dense NF4. So QAT must NOT quantize them: GATEBITS=16.
GATEBITS=16
say "chosen: KVROT=$KVROT GATEBITS=$GATEBITS"
echo "KVROT=$KVROT GATEBITS=$GATEBITS" > logs/leg2_choice.env

# 4. QAT smoke: 25 steps from a COPY of the leg-1 resume file, own tag
cp -p ckpt/resume-c0-long75.pt ckpt/resume-c0-long75-qatsmoke.pt
TAG=c0-long75-qatsmoke STOPAT=7900 KVROT=$KVROT GATEBITS=$GATEBITS \
  bash experiments/run_long_75.sh --go --resume > logs/run-c0-long75-qatsmoke.log 2>&1
RC=$?
grep -E "QAT|eval @|@2048|@8192|step +7900/|STOPPED" logs/run-c0-long75-qatsmoke.log | sed 's/^/[smoke] /'
if [ $RC -ne 0 ] || ! grep -q "STOPPED at step 7900" logs/run-c0-long75-qatsmoke.log; then
  say "QAT smoke FAILED (exit $RC)"; tail -20 logs/run-c0-long75-qatsmoke.log; exit 1
fi
rm -f ckpt/resume-c0-long75-qatsmoke.pt ckpt/resume-c0-long75-qatsmoke-leg1end.pt
say "QAT smoke OK. LEG 2 NOT STARTED (waiting for the user). To start it:"
say "  $(cat logs/leg2_choice.env) nohup bash experiments/run_long_75.sh --go --resume > logs/run-c0-long75-leg2.log 2>&1 &"
