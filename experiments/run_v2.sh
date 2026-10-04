#!/usr/bin/env bash
# v3-absorb (#67, #68, 2026-10-03): THE long run, restarted with (a) the per-head query
# maps fixed (RoPE commutant, after q_norm) and (b) ABSORBABLE MLA -- TransMLA RoRoPE,
# one decoupled exact RoPE key (64/layer) + the CARE latent (4096, water-filled, sequential
# calibration on the training mix), k_norm folded with an exact cached rms: 3.53x cache,
# absorbed decode (#68).
# Same recipe as c0-long75 (#60-#63: GDN-2, ScaleNorm, VeRA-all 1024, plain MLA 75% water-
# filled by the sequential calibration on the training mix, reverse KL + TAID + excess CE
# against the 35B-A3B teacher, text 70.6 / episodes 19.7 / math 9.7% of tokens), plus:
#   --phq-rope-commute --rope-tie-norms   per-head q maps in the RoPE commutant; q/k norm
#       gain changes tied per rotary pair -- c0-long75's unconstrained maps re-routed query
#       content across RoPE frequencies (commutator 0.42-0.53) and long-range multi-key
#       retrieval fell (#66.2).
#   steps 1576+   corpus windows 4096 / 8192 / 32768 at 0.318 / 0.629 / 0.053 (mean 8192:
#       tokens/step unchanged; 32k = 15% of tokens). Steps 1-1575 are leg 1's recipe +
#       the fix only: a clean A/B with c0-long75's step-1575 snapshot.
#   steps 3151+   QAT for the deployed format (#64, #66): NF4 weights, TurboQuant-MSE 4-bit
#       KV latent, bf16 factored gates, NF4 embedding. Evaluated twice at 3150 (before and
#       after installing it) = the PTQ cost on the same weights.
# BUDGET (user: results by Monday): 6300 steps = 40M tokens (CE was flat after 10M in
# c0-long75), OneCycle to zero over them; ~44 h.
# Teacher: the PATCHED llama-server (#65) on 8077, 2 x 33792 slots, -ub 8192, --cache-ram 0
#   (its default 8 GiB host prompt cache is useless here and starved memory, #68.3).
#   bash experiments/run_v2.sh --go            # fresh
#   bash experiments/run_v2.sh --go --resume   # after an interruption
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${1:-}" = "--go" ] || { sed -n '2,24p' "$0"; echo "refusing to start without --go"; exit 1; }
curl -s -m 5 http://127.0.0.1:8077/health | grep -q '"ok"' || { echo "teacher :8077 not healthy"; exit 1; }
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge 45 ] || { echo "only ${AVAIL} GiB available (need >= 45)"; exit 1; }
TAG=${TAG:-v3-absorb}
RES=""
if [ "${2:-}" = "--resume" ]; then
  [ -f "ckpt/resume-$TAG.pt" ] || { echo "no ckpt/resume-$TAG.pt"; exit 1; }
  RES="--resume ckpt/resume-$TAG.pt"
fi
[ -n "${STOPAT:-}" ] && RES="$RES --stop-at-step $STOPAT"
# #68.6 (user, 2026-10-04: finish tonight): end at ANNEALTO with a cosine LR anneal from
# OneCycle's value at ANNEALFROM, TAID ramp completed there, normal final eval / save
[ -n "${ANNEALTO:-}" ] && RES="$RES --anneal-to $ANNEALTO --anneal-from ${ANNEALFROM:?set ANNEALFROM}"
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}   # 20 compile workers held ~8 GiB RSS
exec .venv/bin/python -m mercurius.recovery.train \
  --tag "$TAG" \
  --train-data data/fineweb_edu_long_v2.txt --doc-aware \
  --episodes data/episodes/pilot_mix.decon.jsonl --episode-frac 0.29 \
  --math-data data/math/math_final.decon.jsonl --math-frac 0.16 \
  --gdn2 --scalenorm --train-norms --per-head-q-post --phq-rope-commute --rope-tie-norms \
  --vera-all 1024 --vera-lr 0.01 --lr 3e-05 \
  --mla-calib 256 --mla-calib-seq 1024 --mla-budget 4096 --mla-rope-decouple 1 \
  --divergence reverse --taid-space prob --ce-beta 1.0 \
  --teacher-server http://127.0.0.1:8077 \
  --seed-decay --seed-alpha 0.0 \
  --dial c0 \
  --grad-checkpoint --ckpt-above 0 --seq 8192 \
  --length-mix-spec 4096:0.318,8192:0.629,32768:0.053 --length-mix-start ${LMSTART:-1} \
  --qat --qat-start-step ${QATSTART:-3151} --qat-kv-bits 4 --qat-kv-group 32 --qat-kv-quant tq \
  --qat-kv-rot none --qat-gate-bits 16 --qat-embed-bits 4 \
  --steps ${STEPS:-6300} --eval-every ${EVALEVERY:-1575} --keep-step-ckpts --resume-every 25 \
  --optim nadamw --ema 0 \
  --stuck-prompts data/stuck_prompts_v2.jsonl --stuck-max-new 1024 \
  --mem-cap-gb ${MEMCAP:-40} --log-every ${LOGEVERY:-25} \
  $RES
