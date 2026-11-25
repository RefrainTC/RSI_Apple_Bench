#!/bin/bash
# aa_study.sh <gpu> <label> <n pairs> : parent A/A pairs (+1 warm-up) outside the task results, for per-rep outlier and
# cross-GPU data. Dev-warm caches.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=$1; L=$2; NP=$3; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930; O=$P2/ops_iso14930/study/$L
source $P2/ops_iso14930/guard.sh; mkdir -p $O
stop_check || exit 0; gpu_free $G || exit 3
echo "$(date -Is) aa start $L gpu $G"
GPU=$G python3 -I $H/pairs.py $TD $O/runs st --plan "aa:parent:parent:$NP" --warmup parent --warmup-workload SEED_BASE=1000
echo "$(date -Is) AA DONE"
