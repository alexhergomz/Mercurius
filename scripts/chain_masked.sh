#!/usr/bin/env bash
# The masked-supervision run, and the measurements that make it interpretable.
#
# WHY: 84.7% of episode tokens are tool output. Applying the loss to all of them
# spent twelve times more gradient teaching the model to reproduce file contents
# than to decide what to do -- and the measured failure mode on held-out tasks was
# exactly that it explores correctly and then fails to emit an answer, which lives
# in the 8.4% (decisions D15, docs/agentic_training.md).
#
# Masking BOTH terms, not only CE: every paper doing agentic distillation with a
# KL objective excludes observation tokens from the divergence too
# (arXiv:2505.13820, 2605.07725, 2505.17612).
#
# Run directly, NOT through scripts/guarded.sh: that wrapper backgrounds its
# child and loses its output, so a failure looks like a clean exit with an empty
# log. Bitten by that twice today.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
PY=.venv/bin/python
MIX=data/episodes/pilot_mix.jsonl
COVS=cache/kv_covs_4b_mix.pt
GRP=cache/mla_groups_retr_4096_mix.json

ARCH="--dial nope --seed-decay --seed-alpha 0 --gdn2 --mla-dc 512 --mla-covs $COVS
      --per-head-q --vera-all 1024 --vera-lr 1e-2 --phq-lr 1e-3 --train-norms
      --mla-groups $GRP --scalenorm"
OBJ="--live-teacher --divergence reverse --ce-beta 1.0"
DATA="--doc-aware --train-data data/fineweb_edu_long.txt --episodes $MIX --episode-frac 0.5"

echo "### 1/4  masked run, 150 steps @ 8192"
$PY -m mercurius.recovery.train $ARCH $OBJ $DATA \
  --grad-checkpoint --ckpt-above 0 --seq 8192 --steps 150 \
  --eval-every 75 --log-every 5 --resume-every 25 --mem-cap-gb 60 \
  --lr 3e-5 --tag masked150 > logs/run-masked150.log 2>&1
echo "    exit $? ; $(grep -c '^  step' logs/run-masked150.log) steps logged"
tail -6 logs/run-masked150.log | grep -E "ppl|RECOVERY|@8192" || true

echo "### 2/4  held-out AST tasks, masked arm"
$PY experiments/score_capability.py --arms masked=ckpt/adapters-masked150-best.pt \
  --covs $COVS --mla-groups $GRP --out logs/capability_masked.json \
  > logs/cap_masked.log 2>&1
echo "    exit $?"; grep -E "masked:|pass:|fail:|why:" logs/cap_masked.log | tail -8

echo "### 3/4  held-out AST tasks, UNMASKED arm (the controlled comparison)"
$PY experiments/score_capability.py --arms unmasked=ckpt/adapters-pilotmix-best.pt \
  --covs $COVS --mla-groups $GRP --out logs/capability_unmasked.json \
  > logs/cap_unmasked.log 2>&1
echo "    exit $?"; grep -E "unmasked:|pass:|fail:|why:" logs/cap_unmasked.log | tail -8

echo "### 4/4  standard benchmarks, masked arm"
$PY experiments/bench_standard.py --arms masked=ckpt/adapters-masked150-best.pt \
  --tasks humaneval gsm8k --limit 25 --max-new 700 \
  --covs $COVS --mla-groups $GRP --out logs/bench_masked.json \
  > logs/bench_masked.log 2>&1
echo "    exit $?"; grep -E "humaneval:|gsm8k:" logs/bench_masked.log

echo "### DONE"
