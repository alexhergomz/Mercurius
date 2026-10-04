#!/usr/bin/env bash
# First pilot of the full pipeline on the new corpus.
#
# Architecture is run E's, unchanged, because E is the settled configuration:
# NoPE + GDN-2 + grouped whitened MLA (d_c 512, retrieval-scored groups) +
# per-head query maps + VeRA rank 1024 + ScaleNorm, recovered by reverse-KL
# distillation with TAID and an excess-CE data term. Nothing architectural is
# being tested here.
#
# What IS new is the DATA. The episode mix is the corpus built today, every
# tier provenance-traceable (docs/data_policy.md, docs/decisions.md):
#
#   commit_diff         1,883 seqs  6.16 M tokens  human fixes from 83 permissive
#                                                  repos; prompt is the added
#                                                  TEST, never the commit message
#   verified_retrieval     22 seqs  0.15 M tokens  parser-verified agent
#                                                  trajectories, 4 languages
#   third_party_swe        68 seqs  3.34 M tokens  Nemotron-SWE-v1, CC-BY-4.0,
#                                                  capped at 1 M tokens/repo
#
# The training-time teacher stays the 27B dense in-process, because the
# divergence term needs full logits and the MoE loader does not exist yet (D5).
# The 35B-A3B serves GENERATION on port 8080 at 162 tok/s aggregate; the two
# roles are separate.
#
# The MTP head is OFF. It cost ~38% throughput for a draft that measured 4.9%
# acceptance, and detaching it improved perplexity -- so for a data pilot it is
# noise. Run E's --synth-data is also dropped: the point here is the new corpus.
#
#   scripts/chain_pilot.sh            # smoke then the pilot
#   SMOKE_ONLY=1 scripts/chain_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MIX=data/episodes/pilot_mix.jsonl
[ -s "$MIX" ] || { echo "missing $MIX -- run scripts/build_diff_tasks.py first"; exit 1; }

ARCH="--mla-groups cache/mla_groups_retr_4096.json --scalenorm"
DATA="--episodes $MIX --episode-frac 0.5"

# 3 steps at 2048 first: catches a broken loader or an OOM in a minute rather
# than at step 40 of a long run.
STEPS=3 SEQ=2048 TAG=smoke-pilot scripts/guarded.sh logs/smoke-pilot.log \
  scripts/run_4b_27b.sh $ARCH $DATA \
  --eval-every 1000 --log-every 1 --resume-every 0 --mem-cap-gb 60

if [ "${SMOKE_ONLY:-0}" = "1" ]; then echo "smoke only, stopping"; exit 0; fi

STEPS=150 SEQ=8192 TAG=pilot150 scripts/guarded.sh logs/run-pilot150.log \
  scripts/run_4b_27b.sh $ARCH $DATA \
  --eval-every 50 --mem-cap-gb 60
