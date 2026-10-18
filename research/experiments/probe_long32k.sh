#!/usr/bin/env bash
# #65: memory + throughput of a 32k training step (QAT on, leg-2 config) BEFORE leg 2 --
# 3 steps that are ALL 32k windows, from a COPY of the leg-1 resume file.
#   nohup bash experiments/probe_long32k.sh <boundary pid> > logs/probe_long32k.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PID=${1:?boundary pid}
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
[ -f logs/leg2_choice.env ] || { echo "[probe] no leg2_choice.env -- boundary did not finish"; exit 1; }
. logs/leg2_choice.env
cp -p ckpt/resume-c0-long75.pt ckpt/resume-c0-long75-probe32k.pt
echo "[probe] start $(date '+%a %H:%M:%S')  KVROT=$KVROT GATEBITS=$GATEBITS"
TAG=c0-long75-probe32k STOPAT=7878 LENMIX=32768:1.0 LOGEVERY=1 MEMCAP=${MEMCAP:-60} \
  KVROT=$KVROT GATEBITS=$GATEBITS bash experiments/run_long_75.sh --go --resume \
  > logs/run-c0-long75-probe32k.log 2>&1
RC=$?
grep -E "length mix|step +787[6-8]/|STOPPED|out of memory|Traceback|Error" logs/run-c0-long75-probe32k.log | sed 's/^/[probe] /'
echo "[probe] exit $RC  $(date '+%a %H:%M:%S')"
rm -f ckpt/resume-c0-long75-probe32k.pt ckpt/resume-c0-long75-probe32k-leg1end.pt
