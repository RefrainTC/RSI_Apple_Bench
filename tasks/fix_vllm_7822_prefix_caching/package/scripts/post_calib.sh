#!/bin/bash
# post_calib.sh <task, e.g. iso7822> : the steps the authors ran between a finished calibration and the first scoring
# job (package file: steps 1-3 of the authors' scripts/after_cal.sh, lines copied unchanged apart from the paths; its
# steps 4-5, end-to-end scores and canaries, are RUN.md steps 7-8):
#   1 calibration record re-derived from the runs with the current checks (summarize.py + calib_summary.py)
#   2 offline tasks (OUTREF=1): scored-output thresholds from benign variation (calib_outputs.py), then 1 again.
#     Without them outcheck.py falls back to strict defaults and the human fix of #7822 fails as output_mismatch.
#   3 integrity manifests (scorer, references, venvs) -> $TD/manifest/manifest_hashes.txt
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
H=$ISO_ROOT/harness; T=$1; TD=$ISO_ROOT/tasks/$T; C=$TD/calib
source $TD/task.env
export OUT_MIN_EXACT OUT_TIE_NATS SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE 2>/dev/null
LOG=$C/calibrate.log; echo "$(date '+%F %T') AFTER_CAL START $T" >> $LOG
resum() { python3 -I $H/summarize.py "$TD" "$C/runs" --label cal --calib /dev/null > $C/summary.json && python3 -I $H/calib_summary.py "$C/summary.json" "$C/calib.json" >> $LOG; }
resum || { echo "resummarize failed"; exit 1; }
if [ "${OUTREF:-0}" = 1 ]; then
  python3 -I $H/calib_outputs.py "$TD" >> $LOG 2>&1 || { echo "calib_outputs failed"; exit 1; }
  resum
fi
mkdir -p $TD/manifest
bash $H/integrity.sh make "$TD" > $TD/manifest/manifest_hashes.txt 2>&1 || { echo "integrity make failed"; exit 1; }
cat $TD/manifest/manifest_hashes.txt >> $LOG
python3 -I -c "
import json; c = json.load(open('$C/calib.json'))
print('REF', c['ref_speedup'], 'noise', c['noise_band'], 'witness', (c.get('witness') or {}).get('tolerance_s'), 'thresholds', c.get('outcheck_thresholds', {}).get('tie_nats'), c.get('outcheck_thresholds', {}).get('min_exact'), 'flags', c.get('calibration_flags'))" | tee -a $LOG
echo "$(date '+%F %T') AFTER_CAL DONE $T" >> $LOG
