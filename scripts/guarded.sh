#!/usr/bin/env bash
# Run a GPU job under an external safety monitor that can only ever kill THAT
# job (by PID). Stops it if the GPU reaches KILL_C, system MemAvailable drops
# below MIN_AVAIL_GB, or a second mercurius GPU job appears. Logs one line per
# 3 s to $LOG.safety.
#   scripts/guarded.sh LOGFILE python -m mercurius.eval.ruler ...
set -u
LOG=$1; shift
KILL_C=${KILL_C:-88}; MIN_AVAIL_GB=${MIN_AVAIL_GB:-8}
cd "$(dirname "$0")/.."
export PATH=$PWD/.venv/bin:$PATH
"$@" > "$LOG" 2>&1 &
PID=$!
# GPU jobs only. The pattern used to be any `python -m mercurius...`, which
# counted CPU-only tools too: building the synthetic corpus
# (mercurius.recovery.synth_recall) read as a second GPU job and killed a
# training run at startup on 2026-09-21.
jobs_running() { pgrep -f '^(\.venv/bin/)?python3? (-m mercurius\.(recovery\.train|eval\.)|scripts/thermal_bench)' | wc -l; }
while kill -0 $PID 2>/dev/null; do
  av=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  q=$(timeout 10 nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  t=${q%%,*}; n=$(jobs_running)
  echo "$(date +%T) avail=${av}G gpu=${t}C q=${q} jobs=${n}" >> "$LOG.safety"
  if [ "$av" -lt "$MIN_AVAIL_GB" ] || [ "${t:-0}" -ge "$KILL_C" ] || [ "$n" -gt 1 ]; then
    kill $PID; echo "$(date +%T) KILLED $PID avail=${av}G gpu=${t}C jobs=${n}" >> "$LOG.safety"
  fi
  sleep 3
done
wait $PID; rc=$?; echo "exit $rc" >> "$LOG.safety"; exit $rc
