#!/usr/bin/env bash
# MoL MOVED TO THE FRONT, on request, and the reasoning holds up:
# #45.3 caps every capacity mechanism at <=2.1 points of headroom against a paired
# sigma of ~1.4, so the remaining mechanism arms are chasing noise. Both arms so far
# confirm it -- c0-taps150 came in +0.76 points, z=+0.82, p=0.46, UNRESOLVED.
# MoL's claim is different in KIND: parity at HALF the cache, which is a structural
# saving this eval can actually establish, because the null hypothesis is the one we
# want (accuracy does NOT fall).
#
# TWO ARMS, answering two different questions:
#   mol4-latent  E=4 at the SAME cache as the control (ungrouped 4096, 75%).
#                Does per-token decoder routing buy quality at all? Compare to
#                c0-ungrouped150's 87.2%.
#   mol8-r256    E=8 at HALF the cache (ungrouped 2046, 87.5%). Does routing let us
#                halve the cache for free? #43.2 measured init parity with plain
#                r=512 for exactly this configuration. THIS IS THE ONE THAT MATTERS.
#
# LATENT-ROUTED, COPY INIT, per the user's call for simplicity. Consequences, both
# measured today and both worth stating: the cache is BYTE-IDENTICAL to plain MLA
# because the router reads the cached latent, so no index and no format change
# (#44.1); and with identical-copy init the pairwise cosine between expert GRADIENTS
# is 0.889 (#41.1), so the experts may barely differentiate in 150 steps. If these
# arms come back as exact nulls, that is the most likely reason, and the fix is the
# per-cluster init from collect_cluster_covs.py -- not a different mechanism.
set -uo pipefail
cd "$(dirname "$0")/.."
GU=cache/mla_groups_ungrouped_4096.json
G2=cache/mla_groups_ungrouped_2048.json

run_arm () {   # run_arm <name> <groups> <extra flags...>
  local name="$1" grp="$2"; shift 2
  local ck="ckpt/adapters-c0-$name.pt"
  if [ ! -f "$ck" ]; then
    while pgrep -f "mercurius.recovery.train|bench_full[.]py" >/dev/null; do sleep 60; done
    echo "[q10] === training c0-$name  $(date '+%F %H:%M:%S')"
    bash experiments/run_c0_arm.sh "$name" --mla-groups "$grp" "$@" \
        > "logs/run-c0-$name.log" 2>&1
  fi
  [ -f "$ck" ] || { echo "[q10] FAILED train c0-$name -- see logs/run-c0-$name.log"; return; }
  while pgrep -f "bench_full[.]py" >/dev/null; do sleep 60; done
  echo "[q10] === GSM8K c0-$name  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt --mla-groups "$grp" --dial c0 \
    --arms "c0-$name=$ck" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "c0-$name gsm8k:" logs/bench_gsm8k.log | tail -1
}

# MoL first, half-cache arm second because it is the one with a testable claim
run_arm mol4-latent "$GU" --mla-mol-latent 4
run_arm mol8-r256   "$G2" --mla-mol-latent 8
# then the deprioritised mechanism arms
run_arm gate150     "$GU" --mla-gate xatlu
run_arm f2a2-150    "$GU" --f2a2
echo "[q10] done  $(date '+%F %H:%M:%S')"
