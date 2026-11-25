#!/bin/bash
# n1_queue3.sh : n1 GPU1 is idle after n1a-n1c: long-context transfer at IN=3840 (label tr2) and 3 more A/A pairs.
# Uses the same done/ markers as main_queue2.sh, so main skips whatever n1 already finished.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
D=$OPS/done; mkdir -p $D
step() { local n=$1; shift; [ -e $D/$n ] && { log "n1queue3: skip $n"; return 0; }; stop_check "n1queue3 step $n"
  log "n1queue3: step $n start"; "$@"; local rc=$?; log "n1queue3: step $n rc=$rc"; [ $rc = 0 ] && date -Is > $D/$n; [ $rc = 8 ] && exit 8; return 0; }
step transfer2 bash $OPS/transfer2.sh tr2 1 2 "long3840:IN=3840"
step aa_n1b bash $OPS/aa.sh aa_n1b 1 3
log "n1queue3: done"
