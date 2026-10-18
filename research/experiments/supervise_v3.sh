#!/usr/bin/env bash
# #68.4 supervisor for v3-absorb: keep the teacher up, resume training after any non-zero
# exit (memory-guard stop = 3, crash = other) once memory has recovered; stop on a clean
# finish (exit 0). Detached from the interactive session (setsid) so nothing reaps it.
set -uo pipefail
cd "$(dirname "$0")/.."
say() { echo "[sup] $*  $(date '+%a %H:%M:%S')"; }
teacher_up() { curl -s -m 3 http://127.0.0.1:8077/health | grep -q '"ok"'; }
start_teacher() {
  LLAMA_SPLIT_EMBD_NONE=1 nohup /home/alberto/llama.cpp-split/build/bin/llama-server \
    -m models/qwen3.5-35b-a3b-gguf/Qwen3.5-35B-A3B-Q4_K_M.gguf --embeddings --pooling none \
    --embd-normalize -1 -ngl 99 -c 67584 -b 8192 -ub 8192 --parallel 2 --cache-ram 0 \
    --port 8077 --host 127.0.0.1 >> logs/teacher_8077_split.log 2>&1 &
  echo $! > logs/teacher.pid
  for i in $(seq 120); do teacher_up && break; sleep 5; done
}
for run in $(seq 1 10); do
  if ! teacher_up; then say "teacher down -> starting"; start_teacher; fi
  teacher_up || { say "teacher failed to start"; sleep 120; continue; }
  until [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge 45 ]; do sleep 30; done
  say "resume #$run"
  bash experiments/run_v2.sh --go --resume > logs/run-v3-absorb-$(date +%m%d-%H%M)-s$run.log 2>&1
  rc=$?
  say "trainer exit $rc"
  [ $rc -eq 0 ] && { say "TRAINING FINISHED"; exit 0; }
  sleep 60
done
say "gave up after 10 restarts"
