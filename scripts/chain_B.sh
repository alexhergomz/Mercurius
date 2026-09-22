#!/usr/bin/env bash
# After run A exits: smoke-test run B's exact config for 3 steps, then run B.
# B = baseline + synthetic retention data (2k-8k, ~20% of spans) + conv MTP
# (K=4). On-policy and state passing off.
cd "$(dirname "$0")/.."
while pgrep -f '^(\.venv/bin/)?python3? -m mercurius\.recovery\.train' > /dev/null; do sleep 15; done
B_FLAGS="--synth-data data/synth_recall_4b.txt --mtp 4 --mtp-weight 0.1 --mtp-lr 1e-3"
STEPS=3 SEQ=2048 TAG=smokeB scripts/guarded.sh logs/smoke-B.log scripts/run_4b_27b.sh \
  $B_FLAGS --eval-every 1000 --log-every 1 --resume-every 0 --mem-cap-gb 60
rc=$?; rm -f ckpt/*smokeB*
if [ $rc -ne 0 ] || ! grep -q "mtp " logs/smoke-B.log; then
  echo "SMOKE FAILED rc=$rc"; grep -E "Traceback|Error|STOP|refus" logs/smoke-B.log | tail -3; exit 1
fi
echo "smoke ok: $(grep -E 'synthetic multi-item|conv MTP head|step    3/' logs/smoke-B.log | tr '\n' ' ')"
STEPS=150 SEQ=8192 TAG=4b27b-synthmtp150 scripts/guarded.sh logs/run-B-synthmtp150.log \
  scripts/run_4b_27b.sh $B_FLAGS --eval-every 50 --mem-cap-gb 60
echo "run B exit $?"; grep -E "=== RECOVERY|  @2048:|  @8192:|BEST" logs/run-B-synthmtp150.log
