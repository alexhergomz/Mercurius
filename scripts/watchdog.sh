#!/bin/bash
# Emits ONE LINE PER ACTIONABLE EVENT on stdout; each becomes a notification.
# Covers failure AND completion, because silence must not be ambiguous: a filter
# that only matches success stays quiet through a crash and that looks identical
# to "still running".
#
# Kills by explicit numeric PID only. `pkill -f <pattern>` matches the watchdog's
# own command line and has already killed a wrapper once in this project.
cd "${MERCURIUS_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
KILL_GB=5          # below this, kill the trainer: it cannot checkpoint anyway
WARN_GB=8          # the trainer's own guard: it silently stops saving below this
IDLE_MIN=12        # longer than any legitimate gap between queue steps
warned=0; idle=0; declare -A seen

pids() { ps -eo pid,comm= | awk '$2=="python"{print $1}'; }
freegb() { df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

while true; do
  F=$(freegb); P=$(pids)

  if [ "$F" -lt "$KILL_GB" ]; then
    for p in $P; do kill "$p" 2>/dev/null; done
    echo "STORAGE-KILL free=${F}GB below ${KILL_GB}GB -- killed PIDs: ${P:-none}"
    sleep 120; continue
  elif [ "$F" -lt "$WARN_GB" ] && [ "$warned" = 0 ]; then
    echo "STORAGE-WARN free=${F}GB below the trainer's ${WARN_GB}GB guard -- it will stop writing checkpoints"
    warned=1
  elif [ "$F" -ge "$WARN_GB" ]; then warned=0; fi

  # OOM or crash in any log touched in the last 3 minutes
  for lg in $(find logs -name '*.log' -mmin -3 2>/dev/null); do
    if grep -qiE "out of memory|CUDA error|Killed process|OutOfMemoryError" "$lg"; then
      k="oom:$lg"
      if [ -z "${seen[$k]}" ]; then
        for p in $P; do kill "$p" 2>/dev/null; done
        echo "OOM-KILL in $lg -- killed PIDs: ${P:-none}"
        seen[$k]=1
      fi
    fi
    if grep -qE "^Traceback|FAILED, stopping|not running RULER" "$lg"; then
      k="err:$lg"; [ -z "${seen[$k]}" ] && { echo "FAIL $lg :: $(grep -hE '^[A-Za-z]*Error|FAILED|not running' "$lg" | tail -1 | cut -c1-140)"; seen[$k]=1; }
    fi
    # completions, so silence is never the only signal
    if grep -q "BEST ppl@8192" "$lg"; then
      k="done:$lg:$(grep -c 'BEST ppl@8192' "$lg")"
      [ -z "${seen[$k]}" ] && { echo "DONE $lg :: $(grep 'BEST ppl@8192' "$lg" | tail -1 | cut -c1-120)"; seen[$k]=1; }
    fi
  done

  # DIVERGENCE: a run whose eval perplexity climbs well above where it started
  # is failing, and no crash, OOM or storage signal fires. reverse+ce_beta ran
  # ppl@8192 20.377 -> 23.939 -> 104.972 over 100 steps while this watchdog sat
  # silent, because "still running" and "destroying the model" look identical
  # from the outside.
  for lg in $(find logs -name 'train_*.log' -mmin -6 2>/dev/null); do
    read -r first last n <<<"$(grep -aoE '@8192 +CE [0-9.]+ +ppl +[0-9.]+' "$lg"       | grep -oE 'ppl +[0-9.]+' | grep -oE '[0-9.]+'       | awk 'NR==1{f=$1} {l=$1} END{if(NR)print f, l, NR}')"
    [ -z "$n" ] && continue
    [ "$n" -lt 2 ] && continue
    bad=$(awk -v f="$first" -v l="$last" 'BEGIN{print (l > f*1.25) ? 1 : 0}')
    if [ "$bad" = 1 ]; then
      k="div:$lg:$n"
      if [ -z "${seen[$k]}" ]; then
        for p in $P; do kill "$p" 2>/dev/null; done
        echo "DIVERGE-KILL $lg :: ppl@8192 ${first} -> ${last} over ${n} evals -- killed PIDs: ${P:-none}"
        seen[$k]=1
      fi
    fi
  done

  # idle GPU while queue wrappers are still alive = something is stuck
  if [ -z "$P" ]; then
    idle=$((idle+1))
    if [ "$idle" -ge "$IDLE_MIN" ]; then
      q=$(ps -eo args= | grep -c "[r]un_queue")
      echo "IDLE no python process for ${idle} min; ${q} queue wrapper(s) alive; free=${F}GB"
      idle=0
    fi
  else idle=0; fi
  sleep 60
done
