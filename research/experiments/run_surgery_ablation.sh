#!/usr/bin/env bash
# Locate the ~15-point HumanEval gap to the base (decisions #22). Each arm changes
# ONE thing against ablate-base150; 150 steps because #22 showed HumanEval does not
# move between 150 and 600 steps (65.9% vs 64.6%, sigma ~6 items).
#
#   ablate-base150   restored recipe, 150 steps            <- the reference
#   ablate-uniform150 SAME 4096 budget, UNIFORM 512/layer instead of water-filled.
#                    Isolates allocation POLICY from budget. The heterogeneous plan
#                    squeezes layer 3's [0,1,2] group to 6.9% of its joint maximum
#                    while layer 31's [0] keeps 59.6% -- an 8.6x spread. Rank is a
#                    PERMANENT bottleneck, and the water-filling minimised
#                    reconstruction error AT INIT on the PRE-conversion model, so
#                    it cannot see what a layer needs in order to ADAPT.
#   ablate-mla8094   MLA latent 8094 not 4096 (49% vs 25% of uncompressed),
#                    SAME head grouping, ranks scaled -- only compression differs
#   ablate-rope150   --dial c0 keeps all 32 rotary frequencies instead of NoPE
#
# EMA off in all three: its default start is steps//2 = 75 here and the config was
# wrong anyway (it should run from step 0), so raw final weights keep the arms
# comparable to each other.
#
# SEQUENTIAL because the llama.cpp teacher exposes one slot.
set -uo pipefail
cd "$(dirname "$0")/.."
for tag in base150 uniform150 mla8094 rope150; do
  if [ -f "ckpt/adapters-ablate-$tag.pt" ]; then
    echo "[abl] SKIP $tag: checkpoint already exists"; continue
  fi
  echo "[abl] === training ablate-$tag at $(date '+%F %T')"
  bash "experiments/run_ablate_$tag.sh" > "logs/run-ablate-$tag.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || [ ! -f "ckpt/adapters-ablate-$tag.pt" ]; then
    echo "[abl] FAILED $tag (rc=$rc, checkpoint missing) -- see logs/run-ablate-$tag.log"
    echo "[abl] continuing to the next arm so one failure does not cost the rest"
    continue
  fi
  echo "[abl] done $tag at $(date '+%F %T')"
done

echo "[abl] === benchmarking, ONE INVOCATION PER ARM"
# Each arm MUST be rebuilt with its own groups file and dial. bench_full takes a
# single --mla-groups/--dial for all arms it is given, so passing three arms at
# once would rebuild mla8094 with the 4096 latents (shape mismatch -> strict=False
# drops them -> randomly initialised latents, no error) and rope150 with NoPE
# instead of RoPE (no shape change at all, so nothing complains). Both would
# produce a confident wrong number.
bench () {   # bench <tag> <groups-file> <dial>
  local tag="$1" grp="$2" dial="$3"
  [ -f "ckpt/adapters-ablate-$tag.pt" ] || { echo "[abl] skip bench $tag: no ckpt"; return; }
  echo "[abl] bench $tag  (groups=$(basename "$grp") dial=$dial)"
  .venv/bin/python experiments/bench_full.py \
    --tasks humaneval --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt --mla-groups "$grp" --dial "$dial" \
    --arms "$tag=ckpt/adapters-ablate-$tag.pt" --out logs/bench_ablation.json \
    >> logs/bench_ablation.log 2>&1
  grep -E "$tag humaneval:" logs/bench_ablation.log | tail -1
}
bench base150    cache/mla_groups_retr_4096_mix.json nope
bench uniform150 cache/mla_groups_uniform_4096.json  nope
bench mla8094 cache/mla_groups_retr_8192_mix.json nope
bench rope150 cache/mla_groups_retr_4096_mix.json c0
echo "[abl] ALL DONE $(date '+%F %T')"
