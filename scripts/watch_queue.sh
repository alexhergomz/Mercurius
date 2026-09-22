#!/usr/bin/env bash
# Block silently until the eval queue does something worth reporting, print it,
# and exit: a stage ends (new line in logs/queue.log), a job is KILLED by the
# safety monitor, or a NEW traceback appears in any stage log. Existing content
# at start is ignored, so an old crash does not re-fire.
cd "$(dirname "$0")/.."
count() { [ -f "$1" ] && grep -cE "$2" "$1" || echo 0; }
q0=$(wc -l < logs/queue.log)
declare -A t0 k0
for f in logs/q1_longppl.log logs/q2_ruler_short.log logs/q3_ruler_long.log logs/q4_ruler_256k.log; do
  t0[$f]=$(count "$f" "^Traceback"); k0[$f]=$(count "$f.safety" "KILLED")
done
while true; do
  if [ "$(wc -l < logs/queue.log)" -gt "$q0" ]; then echo "STAGE $(tail -1 logs/queue.log)"; exit 0; fi
  for f in "${!t0[@]}"; do
    [ "$(count "$f" "^Traceback")" -gt "${t0[$f]}" ] && { echo "CRASH in $f: $(grep -E '^[A-Za-z.]*(Error|Exception)' "$f" | tail -1)"; exit 1; }
    [ "$(count "$f.safety" "KILLED")" -gt "${k0[$f]}" ] && { echo "KILLED: $(grep KILLED "$f.safety" | tail -1)"; exit 1; }
  done
  sleep 20
done
