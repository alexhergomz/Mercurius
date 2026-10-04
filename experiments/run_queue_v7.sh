#!/usr/bin/env bash
# ALL UNSETTLED ARCHITECTURE ARMS for model line 1, under --dial c0.
# Control = ablate-rope150 (same recipe, no extras): GSM8K 86.4% (1140/1319).
# Paired sigma on a difference at n=1319 is ~1.4 points, so treat anything under
# ~3 points as unresolved.
#
# Ordered by value. ~45 min train + ~1.8 h GSM8K per arm.
#
#  gate / taps  redone CORRECTLY. Two bugs invalidated #27's versions: (a) they ran
#               at --latent-ext-lr 1e-3 = 33x the dense rate, and both landed at
#               79.3% vs base150's 81.1% -- down_g had travelled to 28% of the
#               on-path latent RMS while nothing co-adapted; the default is now
#               6e-4 = 20x, MatryoshkaKV's validated ratio (#31.4). (b) xatlu.alpha
#               was NOT in _is_x, so it trained in the DENSE group at 3e-5 -- rms
#               3.9e-4 against down_g's 7.0e-3. gate150 therefore tested a barely
#               expanded arctan, not the signed expanded gate it was meant to.
#  conv         NEW, and a different mechanism from taps. Depthwise causal convs on
#               the latent path: pre-down lets the bottleneck compress a temporal
#               WINDOW of the residual stream rather than one position; pre-up mixes
#               cached latents. Shares range(up_k) across lags so the rank ceiling is
#               UNCHANGED (that is taps' axis) -- this buys temporal resolution at
#               k*r params instead of k*r*out. Identity-init, verified exact and
#               causal (reach exactly k, zero backward leakage).
#  mla8094      rank 8094 vs 4096 UNDER RoPE. #28.4: doubling rank was worth +7.9
#               points under NoPE but compression cost only 4.1 in total under RoPE,
#               so the marginal value of rank here may be ~0. If so we ship 4096 and
#               halve the cache. This is a shipping decision, not curiosity.
#  ungrouped    block-diagonal restriction vs shared latent, at identical per-layer
#               rank, under RoPE. The nope version is finishing now.
#  f2a2         head competition, never validly measured: its first benchmark had
#               tau as a non-persistent buffer so rebuilds started with the mask
#               maximal, i.e. the mechanism DISABLED (#22).
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "run_queue_v4[.]sh" >/dev/null; do sleep 60; done

run_arm () {   # run_arm <name> <groups-for-bench> <extra flags...>
  local name="$1" grp="$2"; shift 2
  local ck="ckpt/adapters-c0-$name.pt"
  if [ ! -f "$ck" ]; then
    while pgrep -f "mercurius.recovery.train|bench_full[.]py" >/dev/null; do sleep 60; done
    echo "[q7] === training c0-$name  $(date '+%F %H:%M:%S')"
    bash experiments/run_c0_arm.sh "$name" "$@" > "logs/run-c0-$name.log" 2>&1
  fi
  [ -f "$ck" ] || { echo "[q7] FAILED train c0-$name -- see logs/run-c0-$name.log"; return; }
  while pgrep -f "bench_full[.]py" >/dev/null; do sleep 60; done
  echo "[q7] === GSM8K c0-$name  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt --mla-groups "$grp" --dial c0 \
    --arms "c0-$name=$ck" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "c0-$name gsm8k:" logs/bench_gsm8k.log | tail -1
}

G4=cache/mla_groups_retr_4096_mix.json
G8=cache/mla_groups_retr_8192_mix.json
GU=cache/mla_groups_ungrouped_4096.json

run_arm gate150      "$G4" --mla-gate xatlu
run_arm taps150      "$G4" --mla-taps 1
run_arm conv150      "$G4" --mla-conv 4 --mla-conv-where both
run_arm mla8094-150  "$G8" --mla-groups "$G8"
run_arm ungrouped150 "$GU" --mla-groups "$GU"
run_arm f2a2-150     "$G4" --f2a2
echo "[q7] done  $(date '+%F %H:%M:%S')"
