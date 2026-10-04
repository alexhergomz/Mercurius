#!/usr/bin/env bash
# #68.3: stop v3-absorb right after a resume save, restart the teacher without its 8 GiB
# host prompt cache (--cache-ram 0), resume. Exact PIDs, no pattern kills.
set -uo pipefail
cd "$(dirname "$0")/.."
TRAIN=${1:?trainer pid}; TEACH=${2:?teacher pid}
F=ckpt/resume-v3-absorb.pt
m0=$(stat -c %Y $F)
while [ "$(stat -c %Y $F)" = "$m0" ]; do sleep 5; done
sleep 20                                   # let the atomic rename settle
echo "[rs] resume saved $(date '+%H:%M:%S'); stopping trainer $TRAIN"
kill $TRAIN; while kill -0 $TRAIN 2>/dev/null; do sleep 2; done
echo "[rs] stopping teacher $TEACH"
kill $TEACH; while kill -0 $TEACH 2>/dev/null; do sleep 2; done
LLAMA_SPLIT_EMBD_NONE=1 nohup /home/alberto/llama.cpp-split/build/bin/llama-server \
  -m models/qwen3.5-35b-a3b-gguf/Qwen3.5-35B-A3B-Q4_K_M.gguf --embeddings --pooling none \
  --embd-normalize -1 -ngl 99 -c 67584 -b 8192 -ub 8192 --parallel 2 --cache-ram 0 \
  --port 8077 --host 127.0.0.1 > logs/teacher_8077_split.log 2>&1 &
echo $! > logs/teacher.pid
for i in $(seq 120); do curl -s -m 2 http://127.0.0.1:8077/health | grep -q '"ok"' && break; sleep 5; done
echo "[rs] teacher $(cat logs/teacher.pid) $(curl -s -m2 http://127.0.0.1:8077/health)  $(awk '/MemAvailable/{print int($2/1048576)" GiB avail"}' /proc/meminfo)"
nohup bash experiments/run_v2.sh --go --resume > logs/run-v3-absorb-r2.log 2>&1 &
sleep 5
echo "[rs] trainer resumed: $(ps -eo pid,args | grep '[r]ecovery.train --tag v3-absorb' | awk '{print $1}')  $(date '+%H:%M:%S')"
