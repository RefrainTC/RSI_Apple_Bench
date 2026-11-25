#!/bin/bash
# collect.sh : small text results of iso12380 (< 1 MB each) as a tar.gz on stdout, for the workstation copy.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; TD=$P2/tasks/iso12380; d=$(mkdir -p $P2T/tmp && chmod 700 $P2T/tmp && mktemp -d -p $P2T/tmp col12380.XXXXXX)
trap 'rm -rf "$d"' EXIT
mkdir -p $d/iso12380
cd $P2
{ ls tasks/iso12380/*.py tasks/iso12380/*.sh tasks/iso12380/*.md tasks/iso12380/task.env
  ls tasks/iso12380/calib/{calib.json,calibrate.log,outcheck_calibration.json,summary.json} 2>/dev/null
  ls tasks/iso12380/manifest/{scorer,ref}.sha256 tasks/iso12380/manifest/verify_*.json tasks/iso12380/manifest/made_at.txt 2>/dev/null
  ls tasks/iso12380/results/*/{summary.md,summary.json,score.log,gate.txt,runs_summary.json,changes.patch} 2>/dev/null
  ls tasks/iso12380/results/*/correct/job_meta.json 2>/dev/null
  ls ops_iso12380/*.sh ops_iso12380/logs/* smoke/iso12380_*/smoke_report.json smoke/iso12380_*/smoke.log 2>/dev/null
} | while read -r f; do [ -f "$f" ] && [ $(stat -c %s "$f") -lt 1000000 ] && echo "$f"; done > $d/list
tar -C $P2 -czf - -T $d/list
