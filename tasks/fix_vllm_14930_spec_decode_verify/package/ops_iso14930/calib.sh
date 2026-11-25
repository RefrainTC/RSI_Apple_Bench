#!/bin/bash
# calib.sh <gpu> : full calibration of tasks/iso14930 (5 fix pairs, 5 A/A, 1 venv pair), then output thresholds from the
# trusted runs, re-summary with them, manifests again.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=$1; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930
source $P2/ops_iso14930/guard.sh
stop_check || exit 0; gpu_free $G || exit 3
echo "$(date -Is) calib start gpu $G"
GPU=$G bash $H/calibrate.sh $TD 5 5; echo "$(date -Is) calibrate.sh rc=$?"
GPU=$G python3 -I $P2/ops_iso14930/calib_outputs14930.py $TD 0.05
# package (acceptance fix): re-summarise with task.env loaded, else P2_AUDIT_IGNORE is unset and every pair shows a false audit flag
( source $TD/task.env; export SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE; GPU=$G python3 -I $H/summarize.py $TD $TD/calib/runs --label cal > $TD/calib/summary.json )
python3 -I $H/calib_summary.py $TD/calib/summary.json $TD/calib/calib.json
GPU=$G bash $H/integrity.sh make $TD
echo "$(date -Is) CALIB DONE"
