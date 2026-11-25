#!/bin/bash
# extra_main.sh : after driver_main.sh (main GPU 0): one more fix job with the transfer loads (j4_main, a third transfer
# pair per load) and the K5 batch-shape canary.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
. $ISO_ROOT/ops_iso12380/common.sh
L=$LOGD/driver_$HT.log; FW=$P2/smoke/iso12380_fixws; CR=$P2/smoke/iso12380_canaries
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
until grep -q "driver done" $L; do stop_check; sleep 30; done   # driver_main.sh logs "driver done (<last args>)"
for spec in "j4_$HT $FW --transfer" "canK5_$HT $CR/v12380_K5"; do set -- $spec; stop_check; gpu0_free || exit 3
  [ -d "$2" ] || { log "missing workspace $2"; continue; }
  log "score $1 start"; t0=$(date +%s); bash $H/score.sh $TD $1 $2 ${3:-} > $LOGD/score_$1.out 2>&1
  log "score $1 rc=$? secs=$(( $(date +%s) - t0 )) $(head -3 $TD/results/$1/summary.md 2>/dev/null | tail -2 | tr '\n' ' ')"; done
log "extra done"
