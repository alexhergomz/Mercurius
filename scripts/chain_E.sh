#!/usr/bin/env bash
# After the RULER eval frees the GPU: smoke-test the new MTP arrangement, then
# run E = run D with the heads at full weight, per-head 0.8^j, trunk detached.
cd "$(dirname "$0")/.."
while pgrep -f '^python -m mercurius\.(eval|recovery)' > /dev/null; do sleep 20; done
E="--synth-data data/synth_recall_4b.txt --mtp 4 --mtp-weight 1.0 --mtp-head-decay 0.8 \
   --mla-groups cache/mla_groups_retr_4096.json --scalenorm"
STEPS=3 SEQ=2048 TAG=smokeE scripts/guarded.sh logs/smoke-E.log scripts/run_4b_27b.sh $E \
  --eval-every 1000 --log-every 1 --resume-every 0 --mem-cap-gb 60
rc=$?; rm -f ckpt/*smokeE*
if [ $rc -ne 0 ] || ! grep -q "step    3/3" logs/smoke-E.log; then
  echo "SMOKE E FAILED rc=$rc"; grep -E "Traceback|Error|refus" logs/smoke-E.log | tail -3; exit 1; fi
echo "smoke ok: $(grep 'step    3/3' logs/smoke-E.log | sed -E 's/  \| paced.*//')"
STEPS=150 SEQ=8192 TAG=4b27b-E150 scripts/guarded.sh logs/run-E-mtpfull150.log scripts/run_4b_27b.sh $E \
  --eval-every 50 --mem-cap-gb 60
echo "run E exit $?"; grep -E "  @2048:|  @8192:|BEST" logs/run-E-mtpfull150.log
