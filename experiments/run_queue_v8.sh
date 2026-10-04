#!/usr/bin/env bash
# MIXTURE OF LATENTS arms, under --dial c0. Waits out queue_v7.
#
# Prior art check (arXiv + ar5iv, Sept 2026): NOT FOUND. Every published
# MoE-in-attention design -- MoA 2210.05144, MoH 2410.11842, SwitchHead 2312.07987,
# JetMoE 2404.07413 -- routes the QUERY/OUTPUT side and deliberately keeps K/V
# shared, so the cache is never touched. Routing the K/V decoder is only tractable
# because MLA's absorption moves the E x cost to the query side PER STEP rather
# than per cached key. Nearest existing structure is HydraLoRA 2404.19245 (one
# shared down, several up experts) but for LoRA adapters, soft-mixed, no cache.
# MatryoshkaKV varies the RANK of one basis, not the basis -- different mechanism.
#
# THE ARMS ANSWER ONE QUESTION EACH:
#   mol4-spread  E=4, diverse init. The principled version.
#   mol4-copy    E=4, IDENTICAL-copy init (exact at install). Isolates whether
#                diverse init matters. MoELoRA 2402.12851 predicts copy-init
#                routing stays effectively random, since the router has zero
#                output-based gradient when all experts compute the same thing --
#                a regime nobody has studied. This arm tests that directly.
#   mol2-spread  E=2. The union of expert subspaces is capped at min(E*r, d_out)
#                = 1024 = 2r for our shapes, NOT E*r, because the complement of
#                span(up_k) holds only d_out - r ~ 512 directions. So E=2 already
#                saturates the SPANNING gain; anything E>2 buys is the per-token
#                conditional choice of which r-dim slice to use. This arm separates
#                those two effects.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "run_queue_v7[.]sh" >/dev/null; do sleep 60; done

run_arm () {
  local name="$1"; shift
  local ck="ckpt/adapters-c0-$name.pt"
  if [ ! -f "$ck" ]; then
    while pgrep -f "mercurius.recovery.train|bench_full[.]py" >/dev/null; do sleep 60; done
    echo "[q8] === training c0-$name  $(date '+%F %H:%M:%S')"
    bash experiments/run_c0_arm.sh "$name" "$@" > "logs/run-c0-$name.log" 2>&1
  fi
  [ -f "$ck" ] || { echo "[q8] FAILED train c0-$name -- see logs/run-c0-$name.log"; return; }
  while pgrep -f "bench_full[.]py" >/dev/null; do sleep 60; done
  echo "[q8] === GSM8K c0-$name  $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt \
    --mla-groups cache/mla_groups_retr_4096_mix.json --dial c0 \
    --arms "c0-$name=$ck" --out logs/bench_gsm8k.json >> logs/bench_gsm8k.log 2>&1
  grep -E "c0-$name gsm8k:" logs/bench_gsm8k.log | tail -1
}

# MEASURED 2026-09-27: --mol-spread with random complement directions makes key
# reconstruction WORSE, not better (+0.2% at 0.05, +1.0% at 0.1, +8.5% at 0.3), and
# the reason is structural, not a bad choice of directions: with a SHARED encoder
# c_j holds only the top-r right-singular coordinates, so the discarded energy is
# already destroyed and the least-squares-optimal up_k is UNIQUE. No decoder can
# recover it. Those arms are therefore dropped; a small spread survives only as a
# router symmetry-breaker, which is cheap at 0.05 (+0.2% error).
run_arm mol4-copy    --mla-mol 4 --mol-spread 0.0
run_arm mol4-break   --mla-mol 4 --mol-spread 0.05
run_arm mol2-copy    --mla-mol 2 --mol-spread 0.0
echo "[q8] done  $(date '+%F %H:%M:%S')"
