#!/bin/bash
# resummarize.sh : recompute calibration summary with the task env (audit ignore list, settings ignore) and remake manifests
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930; GPU=${1:-2}
source $TD/task.env; export OUT_MIN_EXACT OUT_TIE_NATS SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE 2>/dev/null
python3 -I $H/summarize.py $TD $TD/calib/runs --label cal > $TD/calib/summary.json
python3 -I $H/calib_summary.py $TD/calib/summary.json $TD/calib/calib.json
for n in ref2 fixc; do echo "correct $n vs ref1: $(python3 -I $TD/compare.py $TD/calib/correct_ref.json $TD/calib/runs/correct_$n/correct.json $CORRECT_MIN_EXACT $CORRECT_TIE_NATS | cut -c1-300)"; done
GPU=$GPU bash $H/integrity.sh make $TD
python3 -I -c "import json;c=json.load(open('$TD/calib/calib.json'));print('flags',c['calibration_flags']);print('audit',c['audit_events_trusted_runs']);print('outcheck',c['outcheck']);print(c['ref_speedup'],c['noise_band'])"
