#!/bin/bash
# v2_n1.sh : on n1 GPU1 after v2_main.sh marked v2_ready: score C3 (v2) and the fix (n1dv2).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done
step() { local n=$1; shift; [ -e $D/$n ] && { log "v2n1: skip $n"; return 0; }; stop_check "v2n1 step $n"
  log "v2n1: step $n start"; "$@"; local rc=$?; log "v2n1: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n; [ $rc = 8 ] && exit 8; return 0; }
until [ -e $D/v2_ready ] || [ -e $P2/STOP_iso13365 ]; do sleep 30; done
step can_C3v2 bash $OPS/score_ws.sh C3v2_window $TD/canaries/C3_window 1
step score_n1dv2 bash $OPS/score_fix.sh n1dv2 1
log "v2n1: done"
