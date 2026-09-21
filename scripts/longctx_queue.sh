#!/usr/bin/env bash
# Long-context evaluation queue: extrapolation + RULER, 4k -> 256k. Each stage
# runs under scripts/guarded.sh; a failed stage is logged and the queue moves on.
cd "$(dirname "$0")/.."
S8=ckpt/adapters-4b27b-s8192-best.pt; S2=ckpt/adapters-4b27b-s2048-best.pt
echo "$(date +%T) stage 1 longppl" >> logs/queue.log
scripts/guarded.sh logs/q1_longppl.log python -m mercurius.eval.longppl \
  --arms orig_bf16=ORIGINAL orig_nf4=ORIGINAL_NF4 s8192=$S8 s2048=$S2 \
  --max-len 262144 --books 4 --quantize --out logs/longppl_256k.json
echo "$(date +%T) stage 1 exit $?; stage 2 ruler 4k-32k" >> logs/queue.log
scripts/guarded.sh logs/q2_ruler_short.log python -m mercurius.eval.ruler \
  --arms orig_nf4=ORIGINAL_NF4 orig_bf16=ORIGINAL s8192=$S8 s2048=$S2 \
  --lengths 4096 8192 16384 32768 --samples 10 --quantize --out logs/ruler_4k_32k.json
echo "$(date +%T) stage 2 exit $?; stage 3 ruler 64k-128k" >> logs/queue.log
T4="niah_single_2 niah_multikey_2 niah_multivalue niah_multiquery"
scripts/guarded.sh logs/q3_ruler_long.log python -m mercurius.eval.ruler \
  --arms orig_nf4=ORIGINAL_NF4 s8192=$S8 --tasks $T4 \
  --lengths 65536 131072 --samples 5 --quantize --out logs/ruler_64k_128k.json
echo "$(date +%T) stage 3 exit $?; stage 4 ruler 256k" >> logs/queue.log
scripts/guarded.sh logs/q4_ruler_256k.log python -m mercurius.eval.ruler \
  --arms orig_nf4=ORIGINAL_NF4 s8192=$S8 --tasks $T4 \
  --lengths 262144 --samples 3 --quantize --out logs/ruler_256k.json
echo "$(date +%T) stage 4 exit $?; queue done" >> logs/queue.log
