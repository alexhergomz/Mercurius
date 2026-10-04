#!/usr/bin/env bash
# Restored masked150 recipe (the 67.7% HumanEval arm) + the three changes asked
# for. GENERATED from logs/recovery-masked150.json's recorded args diffed against
# the parser's own defaults -- not retyped from a log, because decisions #17
# exists precisely because five flags were dropped that way once.
#
# WHY THIS RECIPE. Measured 2026-09-25 on HumanEval, --no-think, max_new 768,
# full 164:
#     original (base)  79.9%
#     masked150        67.7%   <- this recipe
#     control          65.9%   <- same recipe minus vera_lr/train_norms/lr, -1.8
#     swap             52.4%   <- control plus the head swap,          -13.5
# The head swap carries almost all of the deficit, so it is OFF here. The
# learning-rate regressions cost only ~1.8 points, but they are restored anyway
# since they are free.
#
# WHAT IS RESTORED that control/swap had lost:
#   --vera-lr 0.01     adapters at the ADAPTER rate. One group at 2e-4 gave VeRA
#                      a 50x smaller displacement ceiling (lr*steps: 1.5 -> 0.03)
#   --lr 3e-5          dense back to the dense-safe rate, not 2e-4
#   --train-norms      the 64 ScaleNorm scalars + the per-channel norms left out
#                      of the fold. masked150 trained 745 tensors, control 642
#   --scalenorm        folded RMSNorms -> one scalar each
#   --episode-frac 0.5 half the steps on episodes, not 0.3
#   --seed-decay       (--seed-alpha 0.0 with it, as recorded)
#   --mla-dc 512       alongside --mla-groups, as masked150 had both
#   TAID stays ON (it is the parser default; control/swap passed --no-taid)
#
# WHAT IS NEW:
#   --teacher-server   the 30B-class MoE (Qwen3.5-35B-A3B) over llama.cpp,
#                      replacing the in-process 27B. NOTE: masked150 never ran
#                      TAID against the server -- smoke-test before trusting.
#   --steps 150        150 steps was 1,228,800 tokens = 2.9% of the 42.8M-token
#                      corpus. Not data-limited, step-limited. 600 is still only
#                      0.11 epochs, so the old "peaks at 100, degrades by 300"
#                      does NOT apply -- that was 2.5 epochs over a 999k cache.
#   --optim nadamw     torch NAdam + decoupled weight decay. betas held at
#                      (0.9,0.95), NOT NAdam's (0.9,0.999), so the optimizer is
#                      the only change. fp32 state, +196 MiB.
#   --ema 0         0.99^300 = 5% residual on the start point over the ~300
#                      updates after the halfway start. Eval AND checkpoint read
#                      the average, so there is no best-vs-last choice (#20).
#   --stuck-prompts    closed-loop termination monitoring. Diagnostic only.
#
# The terminator/EOS fix needs no flag: it is in the mask, and MASK_VERSION
# invalidates the cached masks (#18).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m mercurius.recovery.train \
  --tag ablate-ungrouped150 \
  --train-data data/fineweb_edu_long.txt \
  --episodes data/episodes/pilot_mix.jsonl --episode-frac 0.5 \
  --doc-aware \
  --gdn2 --scalenorm --train-norms --per-head-q \
  --vera-all 1024 --vera-lr 0.01 \
  --lr 3e-05 \
  --mla-dc 512 --mla-covs cache/kv_covs_4b_mix.pt \
  --mla-groups cache/mla_groups_ungrouped_4096.json \
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
  --log-every 5
