#!/usr/bin/env bash
# STEP-0 ROPE DIAL SWEEP. Usage: run_step0_dial.sh <dial>
#
# WHY. Qwen3.5-4B is ALREADY partial RoPE: head_dim 256 x partial_rotary_factor
# 0.25 = 64 rotary dims per head, built from 32 frequencies. Our "NoPE" did not
# convert a RoPE model -- it zeroed a 64-dim rotary slice the pretrained weights
# were optimised around, and that slice is exactly DeepSeek's qk_rope_head_dim=64.
#
# Measured at step 0, no training:  keep 0 -> 75.4%   keep 32 -> 86.4%  (11.0 pts,
# paired z=9.15). We have only ever measured the two ENDPOINTS. This sweeps between.
#
# DIRECTION MATTERS and my original hypothesis was backwards. The literature says
# keep the HIGH frequencies and drop the LOW ones, not the reverse:
#   p-RoPE (2410.06205): 0.75-RoPE 4.4414 Wiki PPL vs full RoPE 4.4627 vs NoPE
#     4.8594 -- dropping the slowest quarter is free-to-better; NoPE costs ~0.4.
#   HoPE (2410.21216): keeps only theta >= 2pi/L, extrapolates 512->4096 at 13.03
#     PPL while RoPE blows up; in-context copying 23.80% -> 60.23%.
#   MHA2MLA (2502.14837): S_high -0.82% vs S_low -5.25%.
# Our dial's "local" policy keeps the FASTEST-rotating frequencies (inv_freq[0] is
# fastest), i.e. the right end. "global" keeps the slowest -- do not use it.
#
# k4 IS THE KEY ARM: MHA2MLA's own default is r=4 rotary subspaces = 8 dims, and
# they report "increasing dimensionality from 4 to 8 provided negligible
# performance gains". If 4 frequencies recover most of the 11 points, the fix is
# nearly free.
#
# CAVEAT ON SHIPPING. The dial only makes rotations identity; it changes no shapes
# and saves no cache. A nonzero keep means RoPE sits between the rotation and the
# query, which FORBIDS MLA absorption on those dims (TransMLA App. B; the reason
# DeepSeek uses a decoupled head). Full-sequence eval materialises K so this sweep
# measures ACCURACY correctly, but shipping a nonzero keep requires restructuring
# into a shared decoupled RoPE key head. The sweep tells us how many dims to buy
# back; it does not tell us they are free at decode.
set -euo pipefail
DIAL="${1:?usage: run_step0_dial.sh <dial>   (nope|k4|k8|c1|k24|c0)}"
cd "$(dirname "$0")/.."
exec .venv/bin/python -m mercurius.recovery.train \
  --tag "step0-dial-$DIAL" \
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
  --seq 8192 --steps 0 --eval-every 999 \
  --optim nadamw \
  --ema 0 \
  --stuck-prompts data/stuck_prompts_v2.jsonl --stuck-max-new 1024 \
  --mem-cap-gb 24 \
  --dial "$DIAL" \
  --log-every 5
