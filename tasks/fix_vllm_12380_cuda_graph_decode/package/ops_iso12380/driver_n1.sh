#!/bin/bash
# driver_n1.sh [stages] : iso12380 work on the n1 pod's GPU 0 (CPUs 0-7,32-39). Waits for the calibration (main) first.
#   smoke  smoke2.sh on n1 (env check: staged venvs, audit, settings, witness on this host)
#   jobs   independent scoring jobs of the human fix: j1_n1 (--transfer), j2_n1, j3_n1
#   can    canaries K1-K4 (ops_iso12380/make_canaries.sh), one scoring job each
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
. $ISO_ROOT/ops_iso12380/common.sh
L=$LOGD/driver_$HT.log; FW=$P2/smoke/iso12380_fixws; CR=$P2/smoke/iso12380_canaries
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
for st in ${*:-smoke jobs can}; do
  stop_check
  case $st in
    smoke) gpu0_free || exit 3; log "smoke start"; bash $OPS/smoke2.sh > $LOGD/smoke_$HT.out 2>&1; log "smoke rc=$?" ;;
    jobs|can)
      n=0; until [ -f $TD/calib/calib.json ] && grep -q CALIBRATE\ DONE $TD/calib/calibrate.log 2>/dev/null && [ -d $FW ]; do
        stop_check; [ $((n % 10)) = 0 ] && log "waiting for calibration + fix workspace"; n=$((n+1)); sleep 60; done
      if [ $st = jobs ]; then
        for j in "j1_$HT --transfer" "j2_$HT" "j3_$HT"; do set -- $j; stop_check; gpu0_free || exit 3
          [ -f $TD/results/$1/summary.json ] && grep -q speedup $TD/results/$1/summary.json && { log "skip $1 (done)"; continue; }
          log "score $1 start"; t0=$(date +%s); bash $H/score.sh $TD $1 $FW ${2:-} > $LOGD/score_$1.out 2>&1
          log "score $1 rc=$? secs=$(( $(date +%s) - t0 )) $(head -3 $TD/results/$1/summary.md 2>/dev/null | tail -2 | tr '\n' ' ')"; done
      else
        [ -d $CR/v12380_K4 ] || bash $OPS/make_canaries.sh $CR >> $L 2>&1
        for c in K1 K2 K3 K4; do stop_check; gpu0_free || exit 3; lb=can${c}_$HT
          [ -f $TD/results/$lb/summary.json ] && { log "skip $lb (done)"; continue; }
          log "score $lb start"; t0=$(date +%s); bash $H/score.sh $TD $lb $CR/v12380_$c > $LOGD/score_$lb.out 2>&1
          log "score $lb rc=$? secs=$(( $(date +%s) - t0 )) $(head -3 $TD/results/$lb/summary.md 2>/dev/null | tail -2 | tr '\n' ' ')"; done
      fi ;;
  esac
done
log "driver done ($*)"
