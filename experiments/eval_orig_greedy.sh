#!/usr/bin/env bash
# #68.8 the ORIGINAL (NF4 base, bf16 KV) greedy, same settings as v3's final benchmarks --
# the 2026-09-24 original run was sampled (T=0.6, before #50) and is not comparable.
set -uo pipefail
cd "$(dirname "$0")/.."
PID=${1:?v3 bench pid}; while kill -0 $PID 2>/dev/null; do sleep 30; done
say() { echo "[orig] $*  $(date '+%a %H:%M:%S')"; }
for row in "humaneval mbpp:16" "gsm8k:32"; do
  T=${row%%:*}; B=${row#*:}
  say "gen [$T] batch $B"
  .venv/bin/python experiments/bench_full.py --tasks $T --limit 0 --max-new 768 --temperature 0 \
    --no-think --dc 512 --dial c0 --batch $B --arms "ORIG=original" \
    --out logs/gen_greedy_orig.json > "logs/gen_ORIG_${T// /-}.log" 2>&1
  say "gen [$T] exit $?"; grep -E "^  ORIG [a-z0-9]+: " "logs/gen_ORIG_${T// /-}.log" | sed 's/^/[orig] /'
done
say "ALL DONE"
