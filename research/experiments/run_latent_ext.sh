#!/usr/bin/env bash
# The two latent extensions, queued behind whatever is already on the GPU.
#
#   ablate-gate150  --mla-gate xatlu   nonlinear score at FIXED r. Bounded+signed
#                   because the gate output is CACHED: an unbounded gate (swish,
#                   gelu) lets cached values grow without limit, which costs bf16
#                   headroom and breaks later cache quantisation. xATLU's expanded
#                   range is signed, so a token can FLIP a latent direction, not
#                   just attenuate it. Does NOT lift the rank ceiling.
#   ablate-taps150  --mla-taps 1       DOES lift the ceiling: K_j = A_0 c_j + A_1 c_{j-1}
#                   spans up to 2r dimensions instead of r, at ZERO cache cost
#                   since past latents are already stored, and stays absorbable
#                   term by term.
#
# WHY taps=1 SPECIFICALLY. It is the closest thing to a controlled comparison
# against mla8094: that arm doubled effective capacity by DOUBLING THE CACHE
# (4096 -> 8094, 25% -> 49% of uncompressed) and kept 73% of its init advantage.
# taps=1 doubles the key span at NO cache cost. If it matches mla8094 it is a
# strictly better design; if it does not, capacity and cache are not
# interchangeable and that is worth knowing.
#
# Both are zero-init and exactly function-preserving (verified CPU-side), and both
# sit in the --latent-ext-lr group at 1e-3, because zero-init parameters at the
# dense 3e-5 have failed to move five separate times in this codebase.
set -uo pipefail
cd "$(dirname "$0")/.."
WAIT_PID="${1:-}"
if [ -n "$WAIT_PID" ]; then
  echo "[ext] waiting for pid $WAIT_PID to finish before touching the GPU"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
  echo "[ext] pid $WAIT_PID gone at $(date '+%F %H:%M:%S')"
fi
for tag in gate150 taps150; do
  if [ ! -f "ckpt/adapters-ablate-$tag.pt" ]; then
    echo "[ext] === training ablate-$tag $(date '+%H:%M:%S')"
    bash "experiments/run_ablate_$tag.sh" > "logs/run-ablate-$tag.log" 2>&1
    if [ ! -f "ckpt/adapters-ablate-$tag.pt" ]; then
      echo "[ext] FAILED $tag -- see logs/run-ablate-$tag.log; continuing"; continue
    fi
  else
    echo "[ext] SKIP $tag: checkpoint exists"
  fi
  # the extensions wrap the latent's own modules, so the groups file is base150's
  echo "[ext] === GSM8K $tag $(date '+%H:%M:%S')"
  .venv/bin/python experiments/bench_full.py \
    --tasks gsm8k --limit 0 --max-new 768 --temperature 0.7 --no-think \
    --dc 512 --covs cache/kv_covs_4b_mix.pt \
    --mla-groups cache/mla_groups_retr_4096_mix.json --dial nope \
    --arms "$tag=ckpt/adapters-ablate-$tag.pt" --out logs/bench_gsm8k.json \
    >> logs/bench_gsm8k.log 2>&1
  grep -E "$tag gsm8k:" logs/bench_gsm8k.log | tail -1
done
echo "[ext] ALL DONE $(date '+%F %H:%M:%S')"
