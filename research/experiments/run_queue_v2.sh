#!/usr/bin/env bash
# Queue rebuilt in VALUE order after two things changed the priorities:
#
#  * mla8094 gave +4.8 GSM8K points at 3.3 sigma (1133/1319 = 85.9% vs base150's
#    81.1%), closing MORE THAN HALF the 9.4-point gap to the unconverted base.
#    Capacity is the live thread.
#  * the same change was +4 items / 0.5 sigma on HumanEval -- invisible. So the
#    reason I originally dropped rope150 from the GSM8K queue ("its perplexity
#    advantage did not reach HumanEval") was exactly the inference GSM8K just
#    refuted. rope150 had the BEST perplexity of all four arms (9.172).
#
# STEP-0 ARMS ARE THE DENOMINATOR. Every comparison so far has been trained-arm vs
# trained-arm or trained-arm vs base. What recovery BUYS has never been measured,
# so we cannot tell "recovery works and cannot close a surgery-inflicted gap" from
# "recovery is barely doing anything". Per-config step-0 numbers also show whether
# capacity changes what recovery can achieve, not just where it starts.
#
# Ordered best-first so partial completion is still useful.
set -uo pipefail
cd "$(dirname "$0")/.."
bench () {   # bench <tag> <ckpt> <groups> <dial>
  local tag="$1" ck="$2" grp="$3" dial="$4"
  [ -f "$ck" ] || { echo "[q2] skip $tag: $ck missing"; return; }
  echo "[q2] === GSM8K $tag  (groups=$(basename "$grp") dial=$dial)  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt --mla-groups "$grp" --dial "$dial" \
    --arms "$tag=$ck" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "$tag gsm8k:" logs/bench_gsm8k.log | tail -1
}
train () {   # train <script-tag> <ckpt-tag>
  local s="$1" t="$2"
  [ -f "ckpt/adapters-$t.pt" ] && { echo "[q2] SKIP train $t: exists"; return; }
  echo "[q2] === training $t  $(date '+%H:%M:%S')"
  bash "experiments/run_$s.sh" > "logs/run-$t.log" 2>&1
  [ -f "ckpt/adapters-$t.pt" ] || echo "[q2] FAILED $t -- see logs/run-$t.log"
}

# 1. is NoPE the other half of the gap?  (checkpoint already exists)
bench rope150 ckpt/adapters-ablate-rope150.pt cache/mla_groups_retr_4096_mix.json c0

# 2. the denominator: what does recovery buy at all?
train step0_base150 step0-base150
bench step0-base150 ckpt/adapters-step0-base150.pt cache/mla_groups_retr_4096_mix.json nope

# 3. taps: same span as doubling r, at HALF the cache?
train ablate_taps150 ablate-taps150
bench taps150 ckpt/adapters-ablate-taps150.pt cache/mla_groups_retr_4096_mix.json nope

# 4-5. does capacity / NoPE change what recovery ACHIEVES, not just where it starts?
train step0_mla8094 step0-mla8094
bench step0-mla8094 ckpt/adapters-step0-mla8094.pt cache/mla_groups_retr_8192_mix.json nope
train step0_rope150 step0-rope150
bench step0-rope150 ckpt/adapters-step0-rope150.pt cache/mla_groups_retr_4096_mix.json c0

# 6-7. the two least informative: gate is predicted null, allocation is settled
train ablate_gate150 ablate-gate150
bench gate150 ckpt/adapters-ablate-gate150.pt cache/mla_groups_retr_4096_mix.json nope
train ablate_ungrouped150 ablate-ungrouped150
bench ungrouped150 ckpt/adapters-ablate-ungrouped150.pt cache/mla_groups_ungrouped_4096.json nope
echo "[q2] ALL DONE $(date '+%F %H:%M:%S')"
