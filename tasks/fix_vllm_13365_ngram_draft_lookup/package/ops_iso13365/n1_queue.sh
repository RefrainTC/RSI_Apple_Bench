#!/bin/bash
# n1_queue.sh : independent scoring jobs of the human fix on n1 GPU1 (CCD3), after the main pod's calibration exists.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done; mkdir -p $D
step() { local n=$1; shift; [ -e $D/$n ] && { log "n1queue: skip $n"; return 0; }; stop_check "n1 step $n"
  log "n1queue: step $n start"; "$@"; local rc=$?; log "n1queue: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n; [ $rc = 8 ] && exit 8; return 0; }
until [ -e $D/fixws ] || [ -e $P2/STOP_iso13365 ]; do sleep 60; done   # calibration done and fix workspace built on main
stop_check "n1 queue"
for l in n1a n1b n1c; do step score_$l bash $OPS/score_fix.sh $l 1; done
log "n1queue: done"
