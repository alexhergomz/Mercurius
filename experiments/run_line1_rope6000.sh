#!/usr/bin/env bash
# MODEL LINE 1, and the token-budget ablation nobody in this literature has run.
#
# ARCHITECTURE (decision #32.2): GDN-2 lift + MLA r=4096 + the checkpoint's NATIVE
# partial RoPE (--dial c0). Full NoPE is deferred: it is a validated design for
# exactly this architecture (Kimi Linear, KDA:MLA 3:1, kv_lora_rank 512, NoPE on all
# MLA layers, beating its own RoPE variant at 128k) but only FROM SCRATCH at 5.7T
# tokens, and the one post-hoc study (DroPE) needs 2-20B tokens. Both retained
# changes are justified: the GDN-2 lift is EXACT AT INIT and contributed zero of the
# step-0 damage (#28.4), and MLA at 4096 costs only 4.1 GSM8K points once position is
# intact (#28.4) against 15.1 with NoPE.
#
# WHAT IS NEW HERE, versus every previous run:
#   --steps 6000      6000 x ~6000 tok = 36 M tokens, 40x the 0.90 M every arm so far
#                     has had (#30), ~30 h at the measured 336 tok/s. Still 5.5x
#                     short of MatryoshkaKV's 200 M, but it is the first run in this
#                     project not confined to a rounding error of a budget.
#   --synth-data      RESTORES the regression found in #28.6: this flag was live
#                     through control-revkl150/swap-exactkl150, dropped at masked150,
#                     and absent from all 12 generated scripts since -- including
#                     recipe-moe600 and every ablation. 72 MB of synthetic
#                     multi-item-recall documents, ODC-By-derived from FineWeb-Edu.
#   --eval-every 500  a ppl point every 3 M tokens, i.e. a real budget curve.
#   --keep-step-ckpts a checkpoint at EVERY eval, so GSM8K can be measured at
#                     several budgets afterwards. #27 showed ppl and GSM8K disagree,
#                     so the ppl curve alone cannot answer "do more steps help".
#                     12 checkpoints x ~85 MB ~= 1 GB.
#
# WHY THIS RUN IS THE POINT. #27 measured that 150 steps buys ONLY a termination fix:
# on items the model already finished, recovery was +14, +2 and -34 items across
# three arms. The user's reading is that data and steps are the missing ingredient,
# and that a model already 75% NoPE (head_dim 256 x partial_rotary_factor 0.25 = 64
# rotary dims) should be EASIER to transform, not harder. This run tests that
# directly, and no published work has a token-budget curve to compare against.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m mercurius.recovery.train \
  --tag line1-rope6000 \
  --train-data data/fineweb_edu_long.txt \
  --synth-data data/synth_recall_4b.txt \
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
  --seq 8192 --steps 6000 --eval-every 500 \
  --keep-step-ckpts \
  --optim nadamw \
  --ema 0 \
  --stuck-prompts data/stuck_prompts_v2.jsonl --stuck-max-new 1024 \
  --mem-cap-gb 24 \
  --dial c0 \
  --log-every 5
