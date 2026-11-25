#!/bin/bash
# v2_main.sh : correctness v2 (adds the 'far' group, because canary C4 = fix + 256-token search window in the model
# runner passed all v1 checks). Runs on main GPU1 only after every v1 job on both hosts has finished:
# install runner v2 -> recorrect (parent x2 + fix) -> integrity make -> mark v2_ready -> score C4 (v2) and the fix (m4).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done
step() { local n=$1; shift; [ -e $D/$n ] && { log "v2main: skip $n"; return 0; }; stop_check "v2main step $n"
  log "v2main: step $n start"; "$@"; local rc=$?; log "v2main: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n; [ $rc = 8 ] && exit 8; return $rc; }
until { grep -q '\] queue2: done' $OPS/logs/ops.log && grep -q 'n1queue4: done' $OPS/logs/ops.log; } || [ -e $P2/STOP_iso13365 ]; do sleep 30; done
stop_check "v2main"; gpu_free 1 || exit 3
install_v2() {
  A=$TD/v1_archive; mkdir -p $A
  cp -p $TD/runner.py $A/runner_v1.py; cp -p $TD/calib/correct_ref.json $A/correct_ref_v1.json; cp -a $TD/manifest $A/manifest_v1
  cp $OPS/runner_v2.py $TD/runner.py
  sed -i 's/for each of the four prompt groups (repeat, random, late match, small vocabulary)/for each of the prompt groups (repeat, random, late match, small vocabulary; v2 adds far match)/' $TD/compare.py
  grep -q "far" $TD/runner.py && grep -q "v2 adds far" $TD/compare.py
}
step v2_install install_v2 || exit 1
step v2_recorrect env GPU=1 bash $H/recorrect.sh $TD || exit 1
grep -E "recorrect (ref2|fixc) vs ref1" $TD/calib/calibrate.log | tail -2 | cut -c1-300 | while read -r l; do log "v2main: $l"; done
[ "$(grep -E "recorrect (ref2|fixc) vs ref1" $TD/calib/calibrate.log | tail -2 | grep -c '"pass": true')" = 2 ] || { log "v2main: recorrect comparison not clean, stopping"; exit 1; }
step v2_integrity env GPU=1 bash $H/integrity.sh make $TD || exit 1
date -Is > $D/v2_ready
step can_C4v2 bash $OPS/score_ws.sh C4v2_window_rt $TD/canaries/C4_window_rt 1
step score_m4v2 bash $OPS/score_fix.sh m4v2 1
log "v2main: done"
