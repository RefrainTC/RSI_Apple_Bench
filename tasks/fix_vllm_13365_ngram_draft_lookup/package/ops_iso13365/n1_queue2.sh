#!/bin/bash
# n1_queue2.sh : after n1_queue.sh, extra A/A pairs on n1 GPU1, then one more scoring job.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done; mkdir -p $D
step() { local n=$1; shift; [ -e $D/$n ] && { log "n1queue2: skip $n"; return 0; }; stop_check "n1 step $n"
  log "n1queue2: step $n start"; "$@"; local rc=$?; log "n1queue2: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n; [ $rc = 8 ] && exit 8; return 0; }
step aa_n1 bash $OPS/aa.sh aa_n1 1 3
log "n1queue2: done"
