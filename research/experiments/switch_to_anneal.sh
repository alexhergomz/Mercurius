#!/usr/bin/env bash
# #68.6: stop the supervisor, stop the trainer right after a resume save, relaunch the
# supervisor with the anneal (end at 3900). Exact PIDs only.
set -uo pipefail
cd "$(dirname "$0")/.."
SUP=${1:?supervisor pid}; TRAIN=${2:?trainer pid}
kill $SUP; while kill -0 $SUP 2>/dev/null; do sleep 1; done
echo "[sw] supervisor stopped $(date '+%H:%M:%S')"
F=ckpt/resume-v3-absorb.pt; m0=$(stat -c %Y $F)
while [ "$(stat -c %Y $F)" = "$m0" ]; do sleep 5; done; sleep 20
kill $TRAIN; while kill -0 $TRAIN 2>/dev/null; do sleep 2; done
STEP=$(.venv/bin/python -c "import torch;print(torch.load('$F',map_location='cpu',weights_only=False)['step'])")
echo "[sw] trainer stopped after resume save at step $STEP $(date '+%H:%M:%S')"
ANNEALTO=3900 ANNEALFROM=$STEP setsid nohup bash experiments/supervise_v3.sh > logs/supervise_v3b.log 2>&1 < /dev/null &
sleep 3; echo "[sw] supervisor relaunched (anneal 3900 from $STEP)"
