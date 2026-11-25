#!/bin/bash
# poolscreen.sh <gpu> <M> : parent-only pool screen (runner.py POOLSCREEN=M), one job.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=$1; M=$2; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930; O=$P2/ops_iso14930/pool
source $P2/ops_iso14930/guard.sh; mkdir -p $O
stop_check || exit 0; gpu_free $G || exit 3
echo "$(date -Is) start poolscreen M=$M"
GPU=$G bash $H/run_variant.sh $TD parent - "p2_iso14930_pool_g$G" $O/screen_g$G timed POOLSCREEN=$M > $O/screen_g$G.log 2>&1; echo "$(date -Is) end rc=$?"
echo POOL DONE
