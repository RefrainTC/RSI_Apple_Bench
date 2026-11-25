#!/bin/bash
# queue.sh <gpu> <label>:<target>[:transfer] ... : scoring jobs one after another on <gpu>. target "fix" = the human fix
# (calib/oracle_stage, gate bypassed); otherwise a workspace dir under ops_iso14930/ (normal gate).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
G=$1; shift; P2=$ISO_ROOT; TD=$P2/tasks/iso14930
for j in "$@"; do
  L=$(echo $j | cut -d: -f1); T=$(echo $j | cut -d: -f2); F=$(echo $j | cut -d: -f3)
  [ "$T" = fix ] && W=$TD/calib/oracle_stage || W=$P2/ops_iso14930/$T
  [ -e $P2/STOP_iso14930 ] && { echo "$(date -Is) STOP"; exit 0; }
  bash $P2/ops_iso14930/score_job.sh $G $L $W ${F:+--transfer}
done
echo "$(date -Is) QUEUE DONE gpu $G"
