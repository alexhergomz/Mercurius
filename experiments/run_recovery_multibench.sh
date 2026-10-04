#!/usr/bin/env bash
# DOES RECOVERY RECOVER ANYTHING? Asked across four capabilities, not one.
#
# WHY THIS EXISTS. Every recovery conclusion so far rests on GSM8K, and #48 retracted
# the decomposition that claimed recovery does nothing for reasoning. But GSM8K is the
# ONE capability our recovery data does not target: the mix is fineweb-edu long
# documents plus code commit diffs, plus a 72 MB synthetic multi-item-recall corpus
# that has been UNUSED since the #28.6 regression. So if recovery is doing anything,
# the places to look are long-context, retrieval and code. Perplexity already shows an
# effect GSM8K cannot see (control 9.093 against rope150's 9.172).
#
# A null on one benchmark is not a null on capability.
#
# CORRECTNESS NOTE, and the reason this could not have been run earlier: ruler.py and
# longppl.py had NO --dial FLAG and call build(), whose default is dial="nope". Running
# them on a c0 arm would have installed NoPE -- the wrong architecture -- and reported
# a confident number, exactly the hole bench_full.py had. Both now take --dial and it
# is passed explicitly below for every arm. compressed_retrieval.py still hardcodes
# install_rope_dial(m, 0, "global") and is deliberately NOT used here.
#
# ORDER: cheap harnesses first (ruler, longppl are minutes/arm) so partial completion
# still answers something; mbpp last because ~500 generations per arm is hours.
#
# Each pair is step-0 vs 150 steps at the SAME architecture, so the only difference is
# the training. That is the recovery question, stated properly.
set -uo pipefail
cd "$(dirname "$0")/.."
G4=cache/mla_groups_retr_4096_mix.json
G8=cache/mla_groups_retr_8192_mix.json
GU=cache/mla_groups_ungrouped_4096.json
CV=cache/kv_covs_4b_mix.pt
LOG=logs/recovery_multibench.log

wait_gpu () { while pgrep -f "mercurius.recovery.train|bench_full[.]py|mercurius.eval" \
              >/dev/null; do sleep 60; done; }

# ---- config per arm: tag  ckpt  groups  dial -------------------------------
PAIRS=(
  "step0-base150   ckpt/adapters-step0-base150.pt    $G4 nope"
  "base150         ckpt/adapters-ablate-base150.pt   $G4 nope"
  "step0-mla8094   ckpt/adapters-step0-mla8094.pt    $G8 nope"
  "mla8094         ckpt/adapters-ablate-mla8094.pt   $G8 nope"
  "step0-rope150   ckpt/adapters-step0-rope150.pt    $G4 c0"
  "rope150         ckpt/adapters-ablate-rope150.pt   $G4 c0"
  "c0-ungrouped150 ckpt/adapters-c0-ungrouped150.pt  $GU c0"
)

run_one () {   # run_one <harness-label> <tag> <ckpt> <groups> <dial>
  local h="$1" tag="$2" ck="$3" grp="$4" dial="$5"
  [ -f "$ck" ] || { echo "[mb] skip $h/$tag: $ck missing" | tee -a "$LOG"; return; }
  wait_gpu
  echo "[mb] === $h  $tag  (dial=$dial groups=$(basename "$grp"))  $(date '+%F %H:%M:%S')" \
      | tee -a "$LOG"
  case "$h" in
    ruler)
      .venv/bin/python -m mercurius.eval.ruler \
        --arms "$tag=$ck" --dc 512 --covs "$CV" --mla-groups "$grp" --dial "$dial" \
        --lengths 4096 8192 --samples 20 >> "$LOG" 2>&1 ;;
    longppl)
      .venv/bin/python -m mercurius.eval.longppl \
        --arms "$tag=$ck" --dc 512 --covs "$CV" --mla-groups "$grp" --dial "$dial" \
        --books 4 --max-len 32768 >> "$LOG" 2>&1 ;;
    mbpp)
      .venv/bin/python experiments/bench_full.py \
        --tasks mbpp --limit 0 --max-new 768 --temperature 0.7 --no-think \
        --dc 512 --covs "$CV" --mla-groups "$grp" --dial "$dial" \
        --arms "$tag=$ck" --out logs/bench_mbpp.json >> "$LOG" 2>&1 ;;
  esac
  echo "[mb] --- done $h/$tag  $(date '+%H:%M:%S')" | tee -a "$LOG"
}

for h in ruler longppl mbpp; do
  echo "[mb] ######## harness: $h  $(date '+%F %H:%M:%S')" | tee -a "$LOG"
  for row in "${PAIRS[@]}"; do
    set -- $row
    run_one "$h" "$1" "$2" "$3" "$4"
  done
done
echo "[mb] ALL DONE $(date '+%F %H:%M:%S')" | tee -a "$LOG"
