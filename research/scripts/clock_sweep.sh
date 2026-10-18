#!/usr/bin/env bash
# Measure energy per token at several GPU clock caps, then restore the default.
# Run as your normal user (NOT with sudo): it asks for the sudo password once,
# uses root only for nvidia-smi -lgc / -rgc, and runs the benchmark as you.
#   bash ~/mercurius/scripts/clock_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/.."
CLOCKS=${CLOCKS:-"2000 1600 1200"}

sudo -v || exit 1
# keep the sudo ticket alive for the whole sweep (~15-20 min)
( while true; do sudo -n true; sleep 50; done ) & KEEP=$!
restore() { sudo nvidia-smi -rgc > /dev/null; kill $KEEP 2>/dev/null; echo "GPU clocks restored to default"; }
trap restore EXIT

for c in $CLOCKS; do
  echo "=== SM clock capped at $c MHz ==="
  sudo nvidia-smi -lgc 0,"$c" > /dev/null || { echo "could not lock clocks"; exit 1; }
  .venv/bin/python scripts/thermal_bench.py --tag "lgc$c" 2>&1 | grep -E "burst|appended|Error"
done
echo "done -- results in logs/thermal_bench.jsonl"
