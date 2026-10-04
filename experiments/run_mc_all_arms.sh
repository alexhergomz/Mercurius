#!/usr/bin/env bash
# EVERY ablation and mechanism, re-measured with the instrument the field uses (#49).
#
# WHY. All of today's arm comparisons rest on SAMPLED GENERATIVE GSM8K at n=1319,
# paired sigma ~1.4 points on a difference, against only 2.1 points of capacity
# headroom (#45.3) -- so #46.3 could not resolve anything. bench_mc.py is 17,195
# items, DETERMINISTIC (no sampling), averaged over 6 tasks, which is what TransMLA,
# MHA2MLA, CARE, X-EcoMLA, Palu, MOHAWK, LoLCATs, Llamba, SUPRA and DroPE all report.
# It is also ~10x cheaper per arm than GSM8K.
#
# EVERY ARM CARRIES ITS OWN --dial AND --mla-groups. This is not boilerplate: build()
# defaults to dial="nope", so an omitted flag silently evaluates a c0 arm with NoPE
# installed and reports a confident wrong number. That hole existed in bench_full.py,
# ruler.py and longppl.py and was fixed today; getting it wrong here would undo that.
#
# ORIGINAL is included as the unmodified reference -- every published recovery number
# is quoted relative to the base model, and we have never measured ours on this suite.
set -uo pipefail
cd "$(dirname "$0")/.."
CV=cache/kv_covs_4b_mix.pt
G4=cache/mla_groups_retr_4096_mix.json
G8=cache/mla_groups_retr_8192_mix.json
GU=cache/mla_groups_ungrouped_4096.json
GX=cache/mla_groups_uniform_4096.json
LOG=logs/bench_mc.log

ARMS=(
  # tag                ckpt                                  groups dial
  "ORIGINAL            ORIGINAL                               $G4   c0"
  "step0-base150       ckpt/adapters-step0-base150.pt         $G4   nope"
  "base150             ckpt/adapters-ablate-base150.pt        $G4   nope"
  "uniform150          ckpt/adapters-ablate-uniform150.pt     $GX   nope"
  "ungrouped150        ckpt/adapters-ablate-ungrouped150.pt   $GU   nope"
  "gate150             ckpt/adapters-ablate-gate150.pt        $G4   nope"
  "taps150             ckpt/adapters-ablate-taps150.pt        $G4   nope"
  "step0-mla8094       ckpt/adapters-step0-mla8094.pt         $G8   nope"
  "mla8094             ckpt/adapters-ablate-mla8094.pt        $G8   nope"
  "step0-rope150       ckpt/adapters-step0-rope150.pt         $G4   c0"
  "rope150             ckpt/adapters-ablate-rope150.pt        $G4   c0"
  "step0-rope8094      ckpt/adapters-step0-rope8094.pt        $G8   c0"
  "c0-ungrouped150     ckpt/adapters-c0-ungrouped150.pt       $GU   c0"
  "c0-taps150          ckpt/adapters-c0-taps150.pt            $GU   c0"
  "c0-conv150          ckpt/adapters-c0-conv150.pt            $GU   c0"
  "c0-gate150          ckpt/adapters-c0-gate150.pt            $GU   c0"
  "c0-f2a2-150         ckpt/adapters-c0-f2a2-150.pt           $GU   c0"
)

while pgrep -f "mercurius.recovery.train|bench_full[.]py" >/dev/null; do sleep 60; done

for row in "${ARMS[@]}"; do
  set -- $row
  tag="$1"; ck="$2"; grp="$3"; dial="$4"
  if [ "$ck" != "ORIGINAL" ] && [ ! -f "$ck" ]; then
    echo "[mc] skip $tag: $ck missing" | tee -a "$LOG"; continue
  fi
  # resume guard: each arm writes its OWN json, so check THAT file, not a shared
  # one. An earlier version grepped logs/bench_mc.json while the runs wrote to
  # logs/bench_mc_$tag.json, so it never skipped and would redo finished arms.
  if [ -s "logs/bench_mc_$tag.json" ]; then
    echo "[mc] SKIP $tag: logs/bench_mc_$tag.json already present" | tee -a "$LOG"
    continue
  fi
  while pgrep -f "mercurius.recovery.train|bench_full[.]py" >/dev/null; do sleep 60; done
  echo "[mc] === $tag  (dial=$dial groups=$(basename "$grp"))  $(date '+%F %H:%M:%S')" \
      | tee -a "$LOG"
  .venv/bin/python experiments/bench_mc.py \
      --arms "$tag=$ck" --dc 512 --covs "$CV" --mla-groups "$grp" --dial "$dial" \
      --batch 32 --out "logs/bench_mc_$tag.json" >> "$LOG" 2>&1
  grep -E "^  $tag AVERAGE" "$LOG" | tail -1
done
echo "[mc] ALL DONE $(date '+%F %H:%M:%S')" | tee -a "$LOG"
