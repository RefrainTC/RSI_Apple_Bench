#!/bin/bash
# main_queue2.sh : after main_queue.sh on main GPU1: the long-context transfer again at IN=3840 (IN=3968 crashed both
# parent and fix in kv_cache_manager.allocate_slots: prompt + 128 + k draft tokens > max_model_len 4096), then extra A/A.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done; mkdir -p $D
step() { local n=$1; shift; [ -e $D/$n ] && { log "queue2: skip $n"; return 0; }; stop_check "queue2 step $n"
  log "queue2: step $n start"; "$@"; local rc=$?; log "queue2: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n; [ $rc = 8 ] && exit 8; return 0; }
until [ -e $D/score_m3 ] || [ -e $P2/STOP_iso13365 ]; do sleep 60; done
stop_check "queue2"
step transfer2 bash $OPS/transfer2.sh tr2 1 2 "long3840:IN=3840"
step aa_main bash $OPS/aa.sh aa_main 1 3
log "queue2: done"
