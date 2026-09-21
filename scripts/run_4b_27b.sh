#!/usr/bin/env bash
# Qwen3.5-4B student, Qwen3.5-27B teacher, both NF4. Full surgery chain:
# stage A+B (norm fusion, KDA), NoPE, GDN-2, CARE-whitened MLA at d_c=512
# (4.00x KV), per-head query maps; recovery by reverse KL + TAID (prob space,
# adaptive) + excess CE. All trainable weights are adapters or new weights and
# stay bf16/fp32. TAID is on by default (--no-taid to disable).
#
# Heat: lock the GPU clock first (sudo nvidia-smi -lgc 0,2000). Measured on this
# GB10 it sustains ~1.9x the throughput of the default clock at half the power,
# with no thermal pauses at seq 8192 (scripts/clock_sweep.sh).
#
# Prerequisites (once):
#   python scripts/build_stage_ab.py
#   python scripts/save_covs.py --gdn2 --out cache/kv_covs_4b.pt
#   python experiments/get_long_data.py
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-50}
SEQ=${SEQ:-8192}
TAG=${TAG:-4b27b-short}
exec python -m mercurius.recovery.train \
  --dial nope --seed-decay --seed-alpha 0 --gdn2 \
  --mla-dc 512 --mla-covs cache/kv_covs_4b.pt --per-head-q \
  --vera-all 1024 --vera-lr 1e-2 --phq-lr 1e-3 --train-norms \
  --live-teacher --divergence reverse --ce-beta 1.0 \
  --doc-aware --train-data data/fineweb_edu_long.txt \
  --grad-checkpoint --ckpt-above 0 --seq "$SEQ" \
  --steps "$STEPS" --eval-every 25 --log-every 5 --resume-every 25 \
  --lr 3e-5 --tag "$TAG" "$@"
