#!/bin/bash
# recal_post.sh : re-derive the calibration record from the SAME calibration runs after the 2026-11-24 metric change
# (mean -> median rep) and witness change (last long segment). No GPU work. Keeps the first record as calib_v1_mean/.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
. $ISO_ROOT/ops_iso12380/common.sh
source $TD/task.env; export OUT_MIN_EXACT=${OUT_MIN_EXACT:-0} OUT_TIE_NATS SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE 2>/dev/null
C=$TD/calib; V1=$TD/calib_v1_mean; mkdir -p $V1
for f in calib.json summary.json outcheck_calibration.json; do [ -f $V1/$f ] || cp $C/$f $V1/$f; done
cp -a $TD/manifest $V1/manifest_v1 2>/dev/null
rm -f $C/calib.json
python3 -I $H/summarize.py "$TD" "$C/runs" --label cal --calib /dev/null > $C/summary.json
python3 -I $H/calib_summary.py "$C/summary.json" "$C/calib.json"
python3 -I $H/calib_outputs.py "$TD" >> $C/calibrate.log 2>&1
echo "$(date '+%F %T') RE-DERIVED (metric median rep, witness last long segment) from the same runs; v1 record in calib_v1_mean/" >> $C/calibrate.log
# workspace text changed (median): rebuild workspace/pristine (no agent work exists) and the fix workspace
bash $H/build_workspace.sh $TD --force > $LOGD/rebuild_ws.log 2>&1
rm -rf $P2/smoke/iso12380_fixws; bash $H/mk_fix_ws.sh $TD $P2/smoke/iso12380_fixws >> $LOGD/rebuild_ws.log 2>&1
rm -rf $P2/smoke/iso12380_canaries
bash $H/integrity.sh make "$TD" >> $C/calibrate.log 2>&1
echo "$(date '+%F %T') MANIFESTS REMADE after re-derivation" >> $C/calibrate.log
