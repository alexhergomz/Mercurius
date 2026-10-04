#!/usr/bin/env bash
# MODEL LINE 1 architecture arms. ONE latent per layer (NO GROUPS), --dial c0.
#
# WHY UNGROUPED THROUGHOUT (#38.1): at identical per-layer rank, grouped vs shared
# was a clean null -- 117 gains, 109 losses, +0.61 points, z=0.53, p=0.64. Same
# accuracy, same 334 tok/s, one fewer moving part, matches what DeepSeek and
# TransMLA do, and MoL assumes a shared encoder anyway. So the grouped plan is
# retired and CONTROL IS c0-ungrouped150, not ablate-rope150 (which is grouped and
# therefore the wrong reference).
#
# mla8094 IS DROPPED. Two reasons:
#   1. As a GROUPED plan it was degenerate -- single-head groups cap at
#      min(2560, 1*256*2) = 512, and layers 3/15/31 sat at 98.8/97.3/100.0% of full
#      rank. Layer 31 was LITERALLY UNCOMPRESSED. Its earlier +4.8-point win under
#      NoPE was partly just undoing compression on those heads.
#   2. Even ungrouped (1012/layer = 49% of the 2048 cap) it is only 51% cache
#      compression against 75% for 4096 -- a weak target when TransMLA reports 93%.
# MoL replaces it: effective rank WITHOUT cache, which is the actual alternative to
# raising rank rather than a disguised version of it.
#
# Every arm: 150 steps, ~45 min train + ~1.8 h GSM8K. Paired sigma on a difference
# at n=1319 is ~1.4 points, so treat anything under ~3 points as unresolved.
set -uo pipefail
cd "$(dirname "$0")/.."
GU=cache/mla_groups_ungrouped_4096.json
while pgrep -f "run_queue_v4[.]sh" >/dev/null; do sleep 60; done

run_arm () {
  local name="$1"; shift
  local ck="ckpt/adapters-c0-$name.pt"
  if [ ! -f "$ck" ]; then
    while pgrep -f "mercurius.recovery.train|bench_full[.]py" >/dev/null; do sleep 60; done
    echo "[q9] === training c0-$name  $(date '+%F %H:%M:%S')"
    bash experiments/run_c0_arm.sh "$name" --mla-groups "$GU" "$@" \
        > "logs/run-c0-$name.log" 2>&1
  fi
  [ -f "$ck" ] || { echo "[q9] FAILED train c0-$name -- see logs/run-c0-$name.log"; return; }
  while pgrep -f "bench_full[.]py" >/dev/null; do sleep 60; done
  echo "[q9] === GSM8K c0-$name  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt --mla-groups "$GU" --dial c0 \
    --arms "c0-$name=$ck" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "c0-$name gsm8k:" logs/bench_gsm8k.log | tail -1
}

# CONTROL FIRST: everything else is measured against this number.
run_arm ungrouped150
# ALL MoL ARMS REMOVED, on measurement rather than guesswork.
#   mol*-copy   : with identical-copy init and hard top-1 the pairwise cosine between
#                 expert gradients is 0.89, not ~0 -- a random 1/E token subsample
#                 carries essentially the full-data gradient, so experts barely
#                 differentiate. A probable null at 150 steps. --mol-spread does not
#                 help either (0.891 vs 0.889).
#   mol-disjoint: PROVABLY VACUOUS. With oracle routing over disjoint SVD blocks the
#                 reconstruction error equals plain MLA to four decimals (0.2789 vs
#                 0.2789 at layer 3, and at 7/19/31 likewise), because the spectrum
#                 is concentrated and block 0 ALWAYS wins. Selection has nothing to
#                 select.
#   WHAT WORKS   : per-cluster whitened SVD, E=2, ROUTED ENCODER. Oracle-routed
#                 reconstruction error 0.1375 vs 0.2789 (layer 3), 0.1220 vs 0.2564
#                 (7), 0.0833 vs 0.1652 (19), 0.0907 vs 0.1792 (31) -- HALVED, at the
#                 same r + index cache. Diversity must be in the TAIL: every expert
#                 keeps the dominant directions and they differ below them.
#                 Needs a calibration pass (k-means on activations) plus routing the
#                 encoder in transmla.py. NOT YET BUILT -- added when it is.
# capacity / expressiveness, cheapest-signal first
run_arm taps150      --mla-taps 1
run_arm conv150      --mla-conv 4 --mla-conv-where both
run_arm gate150      --mla-gate xatlu
run_arm f2a2-150     --f2a2
echo "[q9] done  $(date '+%F %H:%M:%S')"
