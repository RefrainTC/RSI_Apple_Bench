#!/bin/bash
# score_fix.sh <label> [gpu=1] : one independent scoring job of the human fix (fix workspace through the full gate +
# score.sh: integrity, gate, correctness run, 2 warm-ups, SCORE_PAIRS ABBA pairs, phase 2). Works on main and n1.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
L=$1; G=${2:-1}; stop_check "score $L"; gpu_free $G || exit 3
[ -d $TD/fixws ] || bash $H/mk_fix_ws.sh $TD $TD/fixws || exit 1
log "score $L start gpu=$G $(hostctx)"
t0=$(date +%s); GPU=$G bash $H/score.sh $TD $L $TD/fixws > $LOGD/score_$L.out 2>&1; rc=$?
log "score $L end rc=$rc secs=$(( $(date +%s) - t0 )) $(hostctx) | $(grep -E 'verdict|speedup' $TD/results/$L/summary.md 2>/dev/null | head -2 | tr '\n' ' ')"
