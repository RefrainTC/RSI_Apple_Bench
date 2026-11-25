#!/bin/bash
# collect.sh : bundle the small text results of iso13365 (no outputs.json / correct.json / venvs / weights) into
# ops_iso13365/collect_iso13365.tgz for the workstation copy (results_pod/iso13365/).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; TD=$P2/tasks/iso13365; O=$P2/ops_iso13365
cd $P2
L=$(mktemp -p $P2T/tmp) ; trap 'rm -f $L' EXIT
{
  ls tasks/iso13365/*.py tasks/iso13365/*.sh tasks/iso13365/*.md tasks/iso13365/task.env tasks/iso13365/pristine.sha256
  ls tasks/iso13365/calib/calib.json tasks/iso13365/calib/summary.json tasks/iso13365/calib/calibrate.log tasks/iso13365/calib/oracle_stage/oracle_stage.json 2>/dev/null
  ls tasks/iso13365/manifest/*.sha256 tasks/iso13365/manifest/*.json tasks/iso13365/manifest/*.txt 2>/dev/null | grep -v venv_
  find tasks/iso13365/results -maxdepth 2 -type f \( -name 'summary.*' -o -name 'score.log' -o -name 'gate.txt' -o -name 'changes.patch' -o -name 'integrity_*.json' -o -name 'phase1.json' -o -name 'runs_summary.json' -o -name 'summary.txt' -o -name 'pairs.log' \)
  find tasks/iso13365/calib/runs tasks/iso13365/results tasks/iso13365/smoke tasks/iso13365/n1check -name job_meta.json -o -name passes.json -o -name settings.json -o -name core_counters.jsonl -o -name guard_counts.log 2>/dev/null
  find tasks/iso13365/canaries -maxdepth 1 -mindepth 1 -type d -printf '%p/repo\n' 2>/dev/null | while read d; do (cd $d && git diff > ../canary.diff 2>/dev/null); echo ${d%/repo}/canary.diff; done
  find tasks/iso13365/results_aborted -maxdepth 2 -type f \( -name 'score.log' -o -name 'integrity_*.json' -o -name 'summary.md' \) 2>/dev/null
  ls tasks/iso13365/v1_archive/*.py tasks/iso13365/v1_archive/manifest_v1/*.txt tasks/iso13365/v1_archive/manifest_v1/*.json tasks/iso13365/manifest/verify_*.diff 2>/dev/null
  ls ops_iso13365/*.json ops_iso13365/*.txt 2>/dev/null
  ls harness_iso13365/*.py harness_iso13365/*.sh harness_iso13365/*.md harness_iso13365/audit/*.py
  ls ops_iso13365/*.sh ops_iso13365/*.py ops_iso13365/*.md ops_iso13365/*.log ops_iso13365/logs/*.log ops_iso13365/logs/*.out 2>/dev/null
} | sort -u > $L
tar czf $O/collect_iso13365.tgz -T $L 2>/dev/null; ls -la $O/collect_iso13365.tgz; wc -l < $L
