#!/usr/bin/env bash
# Architecture arms for MODEL LINE 1, i.e. under --dial c0 (native partial RoPE).
# Usage: run_c0_arm.sh <name> [extra train.py flags...]
#
# WHY RE-RUN THINGS WE ALREADY "TESTED". Every 150-step ablation except rope150 ran
# under --dial nope: base150, uniform150, mla8094, taps150, gate150, ungrouped150.
# Decision #32.2 makes line 1 c0, so those arms settled components in a regime we
# have abandoned -- and #28.4 showed the two interact strongly (rank is worth >=1.9x
# more once RoPE is gone). Component choices must be re-measured where they ship.
#
# CONTROL: ablate-rope150 IS this recipe with no extra flags -- GSM8K 1140/1319 =
# 86.4% (n=1319, paired sigma ~1.0 pt, so ~1.4 pt for a difference). Anything here
# is compared against that, not against base150's 81.1% which was a nope arm.
set -euo pipefail
NAME="${1:?usage: run_c0_arm.sh <name> [extra flags...]}"; shift
EXTRA="$*"
cd "$(dirname "$0")/.."
echo "  ARM c0-$NAME   extra: ${EXTRA:-<none, this is the control>}"
exec .venv/bin/python -m mercurius.recovery.train \
  --tag "c0-$NAME" \
  --train-data data/fineweb_edu_long.txt \
  --episodes data/episodes/pilot_mix.jsonl --episode-frac 0.5 \
  --doc-aware \
  --gdn2 --scalenorm --train-norms --per-head-q \
  --vera-all 1024 --vera-lr 0.01 \
  --lr 3e-05 \
  --mla-dc 512 --mla-covs cache/kv_covs_4b_mix.pt \
  --mla-groups cache/mla_groups_retr_4096_mix.json \
  --divergence reverse \
  --teacher-server http://127.0.0.1:8077 \
  --seed-decay --seed-alpha 0.0 \
  --taid-space prob \
  --grad-checkpoint --ckpt-above 0 \
  --seq 8192 --steps 150 --eval-every 999 \
  --optim nadamw \
  --ema 0 \
  --stuck-prompts data/stuck_prompts_v2.jsonl --stuck-max-new 1024 \
  --mem-cap-gb 24 \
  --dial c0 \
  --log-every 5 \
  $EXTRA
