#!/usr/bin/env bash
# Run the F2A2 arm once the control arm finishes, so the two are measured on the
# same machine under the same conditions and differ ONLY by --f2a2.
#
# SEQUENTIAL, not concurrent: the llama.cpp teacher exposes ONE slot, so two runs
# would serialise on it and each would also pay the other's SM contention. No
# wall-clock is saved and the OOM surface doubles.
#
# Keyed on the control's FINAL CHECKPOINT, not just on the process exiting. A
# crashed run also exits, and chaining on exit alone once produced a comparison
# against an arm that never finished.
set -uo pipefail
cd "$(dirname "$0")/.."
CTRL_PID="${1:?usage: chain_f2a2_after_control.sh <control-pid>}"
CTRL_CKPT="ckpt/adapters-recipe-moe600.pt"
echo "[chain] waiting for control pid $CTRL_PID to finish..."
while kill -0 "$CTRL_PID" 2>/dev/null; do sleep 60; done
echo "[chain] control pid $CTRL_PID exited at $(date '+%F %T')"
if [ ! -f "$CTRL_CKPT" ]; then
  echo "[chain] ABORT: $CTRL_CKPT does not exist, so the control did not finish."
  echo "[chain] Not launching the F2A2 arm: a marginal difference against an"
  echo "[chain] unfinished control measures nothing."
  exit 1
fi
echo "[chain] control checkpoint present ($(stat -c%s "$CTRL_CKPT") bytes)"
echo "[chain] launching F2A2 arm at $(date '+%F %T')"
exec bash experiments/run_recipe_moe600_f2a2.sh
