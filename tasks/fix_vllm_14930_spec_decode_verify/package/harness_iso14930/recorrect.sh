#!/bin/bash
# recorrect.sh <taskdir> : regenerate the correctness reference (parent twice + fix once) after a change to the task's
# correctness script, and log both comparisons. Same as the correctness part of calibrate.sh.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
TD=$(readlink -f "$1"); H=$ISO_ROOT/harness_iso14930
source "$TD/task.env"
C=$TD/calib; LOG=$C/calibrate.log; mkdir -p $C/runs
for n in ref1 ref2 fixc; do [ -d "$C/runs/correct_$n" ] && mv "$C/runs/correct_$n" "$C/runs/old_correct_${n}_$(date +%H%M%S)"; done
for t in ref1:parent ref2:parent fixc:fix; do
  n=${t%%:*}; r=${t##*:}
  bash $H/run_variant.sh "$TD" $r - "p2c_${TASK}_recorrect_$n" "$C/runs/correct_$n" correct >> $LOG 2>&1
done
cp "$C/runs/correct_ref1/correct.json" "$C/correct_ref.json"
for n in ref2 fixc; do
  echo "$(date '+%F %T') recorrect $n vs ref1: $(python3 -I $TD/compare.py $C/correct_ref.json $C/runs/correct_$n/correct.json $CORRECT_MIN_EXACT $CORRECT_TIE_NATS)" >> $LOG
done
echo "$(date '+%F %T') RECORRECT DONE" >> $LOG
