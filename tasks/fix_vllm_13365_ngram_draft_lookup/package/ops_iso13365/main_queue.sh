#!/bin/bash
# main_queue.sh : the iso13365 sequence on main GPU1. Each step checks STOP_iso13365 first (and every GPU run checks it
# again in run_variant.sh). Steps already done (marker in ops_iso13365/done/) are skipped, so the queue can be restarted.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done; mkdir -p $D
step() {  # <name> <cmd...>
  local n=$1; shift
  [ -e $D/$n ] && { log "queue: skip $n (done)"; return 0; }
  stop_check "queue step $n"
  log "queue: step $n start"; "$@"; local rc=$?
  log "queue: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n
  [ $rc = 8 ] && exit 8
  return 0
}
step calib bash $OPS/calib.sh 1
[ -f $TD/calib/calib.json ] || { log "queue: no calib.json, stopping"; exit 1; }
step fixws bash $H/mk_fix_ws.sh $TD $TD/fixws
step score_m1 bash $OPS/score_fix.sh m1 1
step transfer bash $OPS/transfer.sh tr1 1 2
step score_m2 bash $OPS/score_fix.sh m2 1
step canaries_make bash $OPS/make_canaries.sh
for c in C1_nopropose C2_k3 C3_window C4_window_rt; do step can_$c bash $OPS/score_ws.sh $c $TD/canaries/$c 1; done
step score_m3 bash $OPS/score_fix.sh m3 1
log "queue: main done"
