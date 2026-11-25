#!/bin/bash
# b3_caldrv.sh <gpu> <pr> <mode:label[:--transfer]> ... : runs calibration / scoring jobs (b3_cal.sh) one after another on
# one GPU of THIS pod. Holds b3/locks/<host>_gpu<gpu>.lock; stages the PR's venvs, model and extras first; before every job
# stops on STOP_batch3 or when the GPU shows any compute process.
G=$1; PR=$2; shift 2; . "$(dirname "$(readlink -f "$0")")/../../env.sh"; P2=$ISO_ROOT; B3=$P2/b3; mkdir -p $B3/locks $B3/logs; HT=$(bash $B3/scripts/hosttag.sh)
L=$B3/logs/caldrv_${HT}_g$G.log; log() { echo "$(date -u '+%F %T') $HT/gpu$G: $*" >> $L; }
exec 9>$B3/locks/${HT}_gpu$G.lock; flock -n 9 || { log "lock busy: not started"; exit 1; }
U=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $G)
. $B3/env/${PR}_cal.env
bash $B3/scripts/b3_stage.sh venv:$B3/venvs/${PR}_parent venv:$B3/venvs/${PR}_fix ${STAGE_EXTRA:-} hf:$M >> $L 2>&1 || { log "staging failed"; exit 1; }
for j in "$@"; do
  IFS=: read mode label tr <<< "$j"
  [ -e $P2/STOP_batch3 ] && { log "STOP_batch3"; break; }
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$U" && { log "GPU busy before $label: stop"; break; }
  log "$PR $mode $label $tr start"; bash $B3/scripts/b3_cal.sh $G $PR $label $mode $tr; rc=$?; log "$PR $mode $label rc=$rc"
  [ $rc = 8 ] && break
done
log "driver end"
