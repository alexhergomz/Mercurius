#!/usr/bin/env bash
# Silent until a training run misbehaves or ends. Exits (one notification) on:
# an MTP loss above 50 nats, an eval ppl@8192 more than 10% above the run's
# step-0 eval, or the run's process ending.
#   scripts/watch_run.sh logs/run-X.log
L=$1
cd "$(dirname "$0")/.."
until [ -s "$L" ] && grep -q "step " "$L"; do sleep 20; done
while pgrep -f '^python -m mercurius\.recovery\.train' > /dev/null; do
  m=$(grep -oE "mtp [0-9.]+" "$L" | awk '$2>50{print $2; exit}')
  p=$(grep -E "^  @8192 " "$L" | grep -oE "ppl +[0-9.]+" | awk '{print $2}')
  p0=$(echo "$p" | head -1); pl=$(echo "$p" | tail -1)
  bad=0; [ -n "$p0" ] && bad=$(awk -v a="$p0" -v b="$pl" 'BEGIN{print (b > a*1.10)}')
  if [ -n "$m" ] || [ "$bad" = 1 ]; then
    echo "ANOMALY in $L: mtp spike=${m:-none} ppl@8192 ${p0}->${pl}"
    grep -E "step |eval @" "$L" | tail -3; exit 1
  fi
  sleep 20
done
echo "run ended (no anomaly): $L"
