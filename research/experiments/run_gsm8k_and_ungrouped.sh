#!/usr/bin/env bash
# GSM8K (n=1319) on the two arms whose effect size matters, then the ungrouped arm.
#
# WHY GSM8K. HumanEval at n=164 has a difference-sigma of 8.3 items (5.1 points),
# so NONE of the five arms measured on 2026-09-26 was distinguishable from another
# -- including the unconverted base (11 items, 1.4 sigma). GSM8K has 1319 items,
# eight times the power: at p~0.85 its difference-sigma is ~12 items = 0.9 points.
# The existing reference run already has base=90.5% and masked150=80.3% there.
#
# ONLY TWO ARMS RE-EVALUATED, deliberately: base150 as the reference and mla8094
# as the one lever that moved both metrics. uniform150 and rope150 are not worth
# the GPU -- allocation is settled (#25) and rope150's perplexity advantage did
# not reach HumanEval at all.
#
# PER-ARM GROUPS FILE IS MANDATORY. bench_full takes one --mla-groups for every arm
# it is given; the wrong file means differently sized latents, strict=False drops
# them, and the arm evaluates with randomly initialised latents and no error.
set -uo pipefail
cd "$(dirname "$0")/.."
gsm () {   # gsm <tag> <ckpt> <groups> <dial>
  local tag="$1" ck="$2" grp="$3" dial="$4"
  [ -f "$ck" ] || { echo "[gsm] skip $tag: $ck missing"; return; }
  echo "[gsm] === $tag  (groups=$(basename "$grp") dial=$dial)  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt --mla-groups "$grp" --dial "$dial" \
    --arms "$tag=$ck" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "$tag gsm8k:" logs/bench_gsm8k.log | tail -1
}
gsm base150 ckpt/adapters-ablate-base150.pt cache/mla_groups_retr_4096_mix.json nope
gsm mla8094 ckpt/adapters-ablate-mla8094.pt cache/mla_groups_retr_8192_mix.json nope

echo "[gsm] === training ablate-ungrouped150 $(date '+%H:%M:%S')"
if [ ! -f ckpt/adapters-ablate-ungrouped150.pt ]; then
  bash experiments/run_ablate_ungrouped150.sh > logs/run-ablate-ungrouped150.log 2>&1
fi
gsm ungrouped150 ckpt/adapters-ablate-ungrouped150.pt cache/mla_groups_ungrouped_4096.json nope
echo "[gsm] ALL DONE $(date '+%F %H:%M:%S')"
