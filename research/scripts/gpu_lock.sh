#!/bin/bash
# Serialise GPU work. Two trainers on a 50W Orin, both hitting an 8192 eval,
# hard-reset the machine on 2026-09-20 03:39 and cost eight hours. A ps check is
# not enough: it races, and an orphaned wrapper can start a run between the check
# and the launch. flock cannot race.
#
#   ./gpu_lock.sh <tag> <command...>
#
# Waits (does not fail) so a queued step runs as soon as the GPU frees.
exec 9>"${TMPDIR:-/tmp}/mercurius-gpu.lock"
if ! flock -w 1 9; then
  echo "[gpu_lock] GPU busy, waiting: $(cat "${TMPDIR:-/tmp}/mercurius-gpu.owner" 2>/dev/null)"
  flock 9
fi
echo "$1 pid $$ since $(date +%H:%M)" > "${TMPDIR:-/tmp}/mercurius-gpu.owner"
shift
"$@"
rc=$?
rm -f "${TMPDIR:-/tmp}/mercurius-gpu.owner"
exit $rc
