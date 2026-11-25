#!/bin/bash
# score_ws.sh <label> <workspace> [gpu=1] : score any workspace (canaries) with score.sh.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
L=$1; WS=$2; G=${3:-1}; stop_check "score $L"; gpu_free $G || exit 3
log "score $L ($WS) start gpu=$G $(hostctx)"
t0=$(date +%s); GPU=$G bash $H/score.sh $TD $L $WS > $LOGD/score_$L.out 2>&1; rc=$?
log "score $L end rc=$rc secs=$(( $(date +%s) - t0 )) | $(grep -E 'verdict|speedup|correctness' $TD/results/$L/summary.md 2>/dev/null | head -3 | tr '\n' ' ')"
