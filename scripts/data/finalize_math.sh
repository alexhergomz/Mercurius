#!/usr/bin/env bash
# Final math data for the long run (#61): stop the procedural teacher job, decontaminate
# the procedural rows written so far, and pack A + B + C + D into
# data/math/math_final.decon.jsonl (the other three sources were decontaminated at row
# level already). Row-level decontamination BEFORE packing, so one flagged problem does
# not drop a whole ~16-problem pack.
set -euo pipefail
cd "$(dirname "$0")/.."
pkill -f "build_math[.]py teacher" || true
# (math is generated on the ONE teacher server, :8077, which stays up for training)
.venv/bin/python experiments/decontam.py --jsonl data/math/procedural_teacher.jsonl \
  --write-clean --out logs/decontam_math_proc.json
PROC=data/math/procedural_teacher.decon.jsonl
[ -f "$PROC" ] || PROC=data/math/procedural_teacher.jsonl   # nothing flagged -> no copy
.venv/bin/python scripts/build_math.py pack \
  data/math/gsm8k_human.decon.jsonl data/math/gsm8k_teacher.decon.jsonl \
  data/math/omi_text.decon.jsonl "$PROC" --out data/math/math_final.decon.jsonl
echo "math packed -> data/math/math_final.decon.jsonl"
