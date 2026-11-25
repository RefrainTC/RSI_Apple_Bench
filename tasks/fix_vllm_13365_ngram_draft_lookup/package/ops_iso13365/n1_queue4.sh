#!/bin/bash
# n1_queue4.sh : canaries on n1 GPU1 while main runs C2/C3: C4 (main skips it via the shared done/ marker), C5, C6.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done; mkdir -p $D
step() { local n=$1; shift; [ -e $D/$n ] && { log "n1queue4: skip $n"; return 0; }; stop_check "n1queue4 step $n"
  log "n1queue4: step $n start"; "$@"; local rc=$?; log "n1queue4: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n; [ $rc = 8 ] && exit 8; return 0; }
C=$ISO_ROOT/tasks/iso13365/canaries
step can_C4_window_rt bash $OPS/score_ws.sh C4_window_rt $C/C4_window_rt 1
step canaries2_make bash $OPS/make_canaries2.sh
step can_C5_nospec bash $OPS/score_ws.sh C5_nospec $C/C5_nospec 1
step can_C6_nmin4 bash $OPS/score_ws.sh C6_nmin4 $C/C6_nmin4 1
log "n1queue4: done"
