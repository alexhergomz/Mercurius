#!/usr/bin/env bash
# THE LONG RUN (line 1) -- recipe settled 2026-09-30 (#60-#62). DOES NOT START WITHOUT --go:
# the user confirms the recipe first.
#
# ARCHITECTURE
#   Qwen3.5-4B student, 24 GDN-2 layers (lifted, exact at init) + 8 full-attention layers
#   converted to PLAIN MLA (no MoL, no conv -- #58.6, #61), one latent per layer
#   (ungrouped), native partial RoPE (--dial c0), ScaleNorm, per-head q maps, ALL-VeRA
#   adapters (rank 1024).
#   KV cache: 4096 latent values / token over the 8 attention layers = EXACTLY 4x (75%,
#   #62.2), water-filled across layers by CARE whitened spectra.
# CALIBRATION (#60): SEQUENTIAL, on the TRAINING MIX, on the fully built student --
#   256 windows x 1024 tokens drawn from the same three sources in the same proportions,
#   covariances re-collected layer by layer from the partly compressed model. Writes a
#   covs file + groups JSON (cache/mla_seqcal_<hash>_*) that every eval harness replays.
# OBJECTIVE (kept, #60): reverse KL + TAID (prob space) against the 35B-A3B teacher
#   (llama.cpp :8077), + ce_beta 1.0 x excess CE on the true token.
# DATA (#61), all decontaminated against every eval set (experiments/decontam.py):
#   text      data/fineweb_edu_long_v2.txt       ~70M tokens, docs >= 8k tokens   70%
#   episodes  data/episodes/pilot_mix.decon.jsonl 9.6M tokens, agentic SWE        20%
#   math      data/math/math_final.decon.jsonl    text CoT only (A-D, #61)        10%
#   token shares -> per-step fractions (mean tokens/draw: text 8192, episodes 4332,
#   math ~3875): episode 0.29, math 0.16, text 0.55 => ~6,350 tokens/step.
# BUDGET: 100M tokens = 15,750 steps at seq 8192; OneCycleLR (5% warmup, peak 3e-5)
#   and TAID span all of it. Eval + kept checkpoint every 1,575 steps (~10M tokens);
#   resume file every 25 steps. ~3.6 days at ~325 tok/s (teacher-bound).
#   USE THE LAST CHECKPOINT (not best-ppl): evals at 10/25/50/100M read the curve.
#
#   bash experiments/run_long_75.sh --go            # after scripts/finalize_math.sh
#   bash experiments/run_long_75.sh --go --resume   # continue after an interruption
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${1:-}" = "--go" ] || { sed -n '2,30p' "$0"; echo; echo "refusing to start without --go"; exit 1; }
MATH=data/math/math_packed_ABC.jsonl
[ -f "$MATH" ] || { echo "missing $MATH -- run scripts/finalize_math.sh first"; exit 1; }
TAG="${TAG:-c0-rt}"
RES=""
[ "${2:-}" = "--resume" ] && RES="--resume ckpt/resume-$TAG.pt"
exec .venv/bin/python -m mercurius.recovery.train \
  --tag "$TAG" \
  --train-data data/fineweb_edu_long_v2.txt --doc-aware \
  --episodes data/episodes/pilot_mix.decon.jsonl --episode-frac 0.29 \
  --math-data "$MATH" --math-frac 0.16 \
  --gdn2 --scalenorm --train-norms --per-head-q \
  --vera-all 1024 --vera-lr 0.01 --lr 3e-05 \
  --mla-calib 256 --mla-calib-seq 1024 --mla-budget 4096 --mla-calib-out cache/mla_seqcal_smoke \
  --divergence reverse --taid-space prob --ce-beta 1.0 \
  --teacher-server http://127.0.0.1:8077 \
  --seed-decay --seed-alpha 0.0 \
  --dial c0 \
  --grad-checkpoint --ckpt-above 0 --seq 8192 \
  --steps 12 --eval-every 999 --pct-start 0.25 --log-every 1 ${STOP:-} --keep-step-ckpts --resume-every 25 \
  --optim nadamw --ema 0 \
   \
  --mem-cap-gb 24  \
  $RES
