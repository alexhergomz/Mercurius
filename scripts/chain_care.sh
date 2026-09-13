#!/usr/bin/env bash
set -u
cd /home/srdelam/qwen-surgery
source env.sh
# wait for the in-flight MLA recovery run, by PID (no pattern matching -- a
# -f match would catch this script's own command line)
if [ -n "" ]; then
  while kill -0 0 2>/dev/null; do sleep 60; done
fi
echo "MLA recovery finished; starting CARE comparison"
./src/diskguard.sh || { echo "disk guard tripped"; exit 1; }
python -u src/run_care.py --calib-samples 256 --calib-seq 512 > logs/care.log 2>&1
echo "CARE exit: $?"
