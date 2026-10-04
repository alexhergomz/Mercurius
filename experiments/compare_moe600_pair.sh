#!/usr/bin/env bash
# Matched closed-loop comparison of the recipe-moe600 pair.
#
# WHY A POST-HOC EVAL RATHER THAN THE IN-TRAINING NUMBERS. The control arm was
# already running when stuck_rate was fixed to report `degenerate`
# (compressibility) instead of only `stuck` (line-novelty, biased low because it
# declines to score ~half the generations and divides by the full count anyway).
# Python had cached the old module, so the control's six evals all used the old
# metric while the F2A2 arm, starting fresh, uses both. The in-training numbers
# are therefore NOT comparable across the pair. This re-measures both finished
# checkpoints with one code version, one cap, one seed.
#
# CAP 6144, not the 1024 used during training, for two reasons: it is what
# control-revkl150 / hidden / swap were measured at on 2026-09-25, so all five
# arms land on one axis; and at 1024 the earlier arms were censored on 17-31% of
# generations, which is the regime "Mind the Cap" (2608.04160) shows can reverse
# rankings.
#
# --stuck-dump so the generations are READABLE. Every closed-loop conclusion that
# survived today came from reading text; every one that came from a rate alone
# had to be retracted.
#
# NOTE --f2a2 on the second arm is REQUIRED, not cosmetic: wrapping o_proj
# renames the parameters under it (o_proj.base.vera_d -> o_proj.o_proj.base.vera_d),
# so without it every o_proj adapter is unmatched and strict=False drops them in
# silence -- the arm would score worse for a reason nothing in the log explains.
set -uo pipefail
cd "$(dirname "$0")/.."
COMMON=(--doc-aware --train-data data/fineweb_edu_long.txt
        --episodes data/episodes/pilot_mix.jsonl --episode-frac 0.5
        --gdn2 --scalenorm --train-norms --per-head-q
        --vera-all 1024 --vera-lr 0.01 --lr 3e-05
        --mla-dc 512 --mla-covs cache/kv_covs_4b_mix.pt
        --mla-groups cache/mla_groups_retr_4096_mix.json
        --divergence reverse --teacher-server http://127.0.0.1:8077
        --seed-decay --seed-alpha 0.0 --taid-space prob
        --grad-checkpoint --ckpt-above 0 --seq 8192 --steps 0
        --mem-cap-gb 32 --stuck-only
        --stuck-prompts data/stuck_prompts_v2.jsonl
        --stuck-max-new 6144 --stuck-seed 0)

# RAW vs EMA is evaluated too, because the step-400 eval showed ppl improving
# (9.420 -> 9.354) while termination REGRESSED (hit cap 2% -> 12%, novel-tail
# 1.00 -> 0.84) at exactly the step averaging switched on, with the stuck seed
# fixed so only the weights differed. If the average is a worse generator, the
# shipped checkpoint is the wrong one -- eval and save both read the average.
# The resume file carries the RAW step-600 iterate (written under
# ema.suspended()), so the pair is measurable with no extra training.
for arm in control f2a2 control-raw; do
  case "$arm" in
    control)
      CK=ckpt/adapters-recipe-moe600.pt; EXTRA=(); TAG=MOE600 ;;
    f2a2)
      CK=ckpt/adapters-recipe-moe600-f2a2.pt; EXTRA=(--f2a2); TAG=MOE600-F2A2 ;;
    control-raw)
      # unwrap the resume payload: it nests the tensors under "weights"
      RS=ckpt/resume-recipe-moe600.pt
      CK=/tmp/claude-1000/_raw_moe600.pt; EXTRA=(); TAG=MOE600-RAW
      if [ -f "$RS" ]; then
        .venv/bin/python - "$RS" "$CK" <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
w = sd["weights"] if "weights" in sd else sd
torch.save(w, sys.argv[2])
print(f"  [compare] unwrapped raw iterate at step {sd.get('step')}: {len(w)} tensors")
PYEOF
      fi ;;
  esac
  if [ ! -f "$CK" ]; then
    echo "[compare] SKIP $arm: $CK missing (arm did not finish)"; continue
  fi
  echo "[compare] === $TAG from $CK"
  .venv/bin/python -m mercurius.recovery.train "${COMMON[@]}" "${EXTRA[@]}" \
    --stuck-ckpt "$CK" --stuck-dump "logs/gens_${TAG}.jsonl" --tag "$TAG"
done
echo "[compare] DONE"
