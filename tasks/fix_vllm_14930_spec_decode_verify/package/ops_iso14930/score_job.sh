#!/bin/bash
# score_job.sh <gpu> <label> <stage dir|workspace dir> [--transfer] : one independent scoring job (score.sh) on <gpu>.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=$1; L=$2; W=$3; TR=${4:-}; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930
source $P2/ops_iso14930/guard.sh
stop_check || exit 0; gpu_free $G || exit 3
case "$W" in */oracle_stage|*/stages/*) WS="STAGE:$W";; *) WS=$W;; esac
echo "$(date -Is) score start $L gpu $G $WS $TR"
GPU=$G bash $H/score.sh $TD $L "$WS" $TR; echo "$(date -Is) score end $L rc=$?"
