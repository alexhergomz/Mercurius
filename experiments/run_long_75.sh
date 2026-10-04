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
# TWO LEGS of ~1.8 days (user, 2026-09-30), ONE schedule: LR, TAID and the sampler span
# all 15,750 steps; leg 1 stops cleanly at step 7,875 (= 5 x 1,575, an eval step) and
# writes the resume file there; leg 2 continues from it to the end. Resume verified
# (#63): a stop-at-6 + resume reproduces an uninterrupted run up to the run-to-run GPU
# noise that two UNINTERRUPTED runs also show. Resume also works after an unplanned
# interruption (resume file every 25 steps, written atomically).
#
#   bash experiments/run_long_75.sh --go            # leg 1: steps 1-7875, then stops
#   KVROT=none GATEBITS=16 bash experiments/run_long_75.sh --go --resume   # leg 2 (QAT): steps 7876-15750
#   bash experiments/run_long_75.sh --go --resume-leg1  # leg 1 interrupted: resume, stop at 7875
# NEVER change the data files between legs: the resume refuses on a data fingerprint
# mismatch.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "${1:-}" = "--go" ] || { sed -n '2,30p' "$0"; echo; echo "refusing to start without --go"; exit 1; }
# PRE-FLIGHT: the teacher must be up (a dead teacher stops training), and nothing heavy may
# share the machine -- an idle 8077 teacher was OOM-killed once while two 35B servers and
# eval batches ran together. ONE teacher serves everything: llama-server --embeddings
# --pooling none --embd-normalize -1 -ngl 99 -c 67584 -b 16384 -ub 16384 --parallel 8
# (8 slots x 8448 tokens, each >= seq + on-policy margin 8256). LEG 2 (#65, 32k windows):
# -c 67584 --parallel 2 -b 33792 -ub 33792 (2 slots x 33792 >= 32768 + 64). NEVER generate on it while
# training: continuous generation STARVES the long hidden-state requests (measured
# 2026-09-30: a training probe waited >20 min behind 6 generation slots and went through
# within a minute of pausing them). scripts/finalize_math.sh stops math generation.
curl -s -m 5 http://127.0.0.1:8077/health | grep -q '"ok"' || { echo "teacher :8077 not healthy"; exit 1; }
AVAIL=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$AVAIL" -ge 45 ] || { echo "only ${AVAIL} GiB available (need >= 45): stop other GPU jobs first"; exit 1; }
MATH=data/math/math_final.decon.jsonl
[ -f "$MATH" ] || { echo "missing $MATH -- run scripts/finalize_math.sh first"; exit 1; }
TAG=${TAG:-c0-long75}                          # override ONLY for smokes on a copy
RES="--stop-at-step 7875"                      # leg 1
if [ "${2:-}" = "--resume" ] || [ "${2:-}" = "--resume-leg1" ]; then
  [ -f "ckpt/resume-$TAG.pt" ] || { echo "no ckpt/resume-$TAG.pt to resume"; exit 1; }
  RES="--resume ckpt/resume-$TAG.pt"          # leg 2 / recovery: run to the end
  # an UNPLANNED interruption during leg 1: resume but still stop at the leg boundary
  if [ "${2:-}" = "--resume-leg1" ]; then
    RES="$RES --stop-at-step 7875"
  else
    # LEG 2 = QAT (#64, user 2026-10-01: deploy 4-bit weights + 4-bit KV). The
    # deployed format is NF4 weights (VeRA merged, maps/rotation folded), GDN-2
    # gates in their exact factored bf16 form (#64.1, so unquantized: GATEBITS=16), int4 KV latent g32, NF4 tied embedding -- src/mercurius/models/qat.py.
    # KVROT comes from experiments/qat_ptq_sweep.py, run on the leg-1 adapters
    # first. Leg 1's endpoint is kept: leg 2 overwrites resume-$TAG.pt from its
    # first eval on, and a different leg 2 must remain startable from it.
    # (also covers a leg-2 crash: a re-run resumes from the latest leg-2 file,
    # and the leg-1 copy is only made once)
    [ -f "ckpt/resume-$TAG-leg1end.pt" ] || cp -p "ckpt/resume-$TAG.pt" "ckpt/resume-$TAG-leg1end.pt"
    RES="$RES --qat --qat-kv-bits 4 --qat-kv-group 32 --qat-kv-rot ${KVROT:-none} --qat-kv-quant ${KVQ:-tq} --qat-gate-bits ${GATEBITS:-16} --qat-embed-bits 4"
    # LONG WINDOWS (#65, user 2026-10-02): corpus windows become a length mix with the
    # SAME mean (8192), so tokens/step, total tokens, step count and source shares are
    # all unchanged; 32k windows carry 15.0% of all tokens. Needs the teacher with
    # >= 32832-token slots: llama-server ... -c 67584 --parallel 2 -b 33792 -ub 33792.
    # The stock llama.cpp build cannot return hidden states for a prompt above ~12k
    # tokens in ONE micro-batch (14336 crashes the CUDA MoE path, >=16k asserts in
    # mmid.cu), so the teacher is the PATCHED server (#65; ~/llama.cpp-split,
    # LLAMA_SPLIT_EMBD_NONE=1): pooling-none prompts split into 8k micro-batches
    # through the KV cache, per-token outputs concatenated. Verified at the kernels'
    # own noise floor (experiments/check_teacher_split.py: KL 0.0017 vs floor 0.0014).
    #   LLAMA_SPLIT_EMBD_NONE=1 ~/llama.cpp-split/build/bin/llama-server -m ... --embeddings
    #     --pooling none --embd-normalize -1 -ngl 99 -c 67584 -b 8192 -ub 8192 --parallel 2
    RES="$RES --length-mix-spec ${LENMIX:-4096:0.318,8192:0.629,32768:0.053}"
  fi
fi
[ -n "${STOPAT:-}" ] && RES="$RES --stop-at-step $STOPAT"   # smokes: stop early
exec .venv/bin/python -m mercurius.recovery.train \
  --tag "$TAG" \
  --train-data data/fineweb_edu_long_v2.txt --doc-aware \
  --episodes data/episodes/pilot_mix.decon.jsonl --episode-frac 0.29 \
  --math-data "$MATH" --math-frac 0.16 \
  --gdn2 --scalenorm --train-norms ${PHQ---per-head-q} \
  --vera-all 1024 --vera-lr 0.01 --lr 3e-05 \
  --mla-calib 256 --mla-calib-seq 1024 --mla-budget 4096 \
  --divergence reverse --taid-space prob --ce-beta 1.0 \
  --teacher-server http://127.0.0.1:8077 \
  --seed-decay --seed-alpha 0.0 \
  --dial c0 \
  --grad-checkpoint --ckpt-above 0 --seq 8192 \
  --steps 15750 --eval-every 1575 --keep-step-ckpts --resume-every 25 \
  --optim nadamw --ema 0 \
  --stuck-prompts data/stuck_prompts_v2.jsonl --stuck-max-new 1024 \
  --mem-cap-gb ${MEMCAP:-40} --log-every ${LOGEVERY:-25} \
  $RES
