#!/bin/bash
# driver_main.sh [stages] : iso12380 work on the main pod's GPU 0 (CPUs 0-7,32-39). Stages (default all, in order):
#   cal   calibrate.sh 5 AB + 5 A/A (+ overlay-equivalence pair), output thresholds, manifests
#   fixws the human-fix workspace for scoring (smoke/iso12380_fixws)
#   jobs  independent scoring jobs of the human fix: j1_main (--transfer), j2_main, j3_main
#   pw    the pristine workspace scored (expect ~1.0, correctness pass)
# Each GPU job checks STOP_iso12380 first and refuses when anything else is on GPU 0.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
. $ISO_ROOT/ops_iso12380/common.sh
L=$LOGD/driver_$HT.log; FW=$P2/smoke/iso12380_fixws
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
for st in ${*:-cal fixws jobs pw}; do
  stop_check
  case $st in
    cal) gpu0_free || exit 3; log "calibrate start"; bash $H/calibrate.sh $TD 5 5 > $LOGD/calibrate_$HT.out 2>&1; log "calibrate rc=$?" ;;
    fixws) [ -d $FW ] || bash $H/mk_fix_ws.sh $TD $FW >> $L 2>&1; log "fixws $(ls $FW/repo/vllm | wc -l) entries" ;;
    jobs) for j in "j1_$HT --transfer" "j2_$HT" "j3_$HT"; do set -- $j; stop_check; gpu0_free || exit 3
            [ -f $TD/results/$1/summary.json ] && grep -q speedup $TD/results/$1/summary.json && { log "skip $1 (done)"; continue; }
            log "score $1 start"; t0=$(date +%s); bash $H/score.sh $TD $1 $FW ${2:-} > $LOGD/score_$1.out 2>&1
            log "score $1 rc=$? secs=$(( $(date +%s) - t0 )) $(head -3 $TD/results/$1/summary.md 2>/dev/null | tail -2 | tr '\n' ' ')"; done ;;
    pw) stop_check; gpu0_free || exit 3; log "score pw_$HT start"; t0=$(date +%s)
        bash $H/score.sh $TD pw_$HT $TD/workspace > $LOGD/score_pw_$HT.out 2>&1
        log "score pw_$HT rc=$? secs=$(( $(date +%s) - t0 )) $(head -3 $TD/results/pw_$HT/summary.md 2>/dev/null | tail -2 | tr '\n' ' ')" ;;
  esac
done
log "driver done ($*)"
