#!/bin/bash
# recorrect_job.sh <gpu> <label> : re-run only the correctness run of a scoring job whose correctness run crashed for an
# infrastructure reason (EADDRINUSE on the engine's TCPStore port, 2026-11-24 s1_g2), then phase 2. Same stage, same GPU,
# same lock as score.sh. The crashed run is kept as correct.crashed/.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=$1; L=$2; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930; R=$TD/results/$L
source $P2/ops_iso14930/guard.sh
exec 9>"$P2/locks/gpu$G.lock"; flock -w 7200 9 || exit 3
stop_check || exit 0; gpu_free $G || exit 3
source $TD/task.env; export P2_CACHE_MODE=job P2_CACHE_ROOT=$P2T/cache/$TASK/job-$L-recorrect-$(date +%s); trap 'rm -rf "$P2_CACHE_ROOT"' EXIT
[ -d $R/correct.crashed ] || mv $R/correct $R/correct.crashed
echo "$(date '+%F %T') RECORRECT (EADDRINUSE crash of the first correctness run)" >> $R/score.log
GPU=$G bash $H/run_variant.sh $TD cand $R/snap/stage "p2s_${TASK}_${L}_correct2" $R/correct correct >> $R/score.log 2>&1
GPU=$G bash $H/score_phase2.sh $TD $L
