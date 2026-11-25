#!/bin/bash
# snapshot_gate.sh <taskdir> <workspace> <dest dir> : copy the workspace to <dest>/ws (tar; excluding dev_runs, .git, __pycache__,
# *.pyc), run kernel_gate (vs pristine), lock_check and the P2 diff rules (p2_gate_rules.py). On pass: <dest>/stage (overlay + deletions), <dest>/changes.patch.
# exit 0 pass / 1 reject / 2 error. Writes <dest>/gate.json, <dest>/lock.json, <dest>/gate.txt.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
TD=$(readlink -f "$1"); WS=$(readlink -f "$2"); D=$3; H=$ISO_ROOT/harness_iso13365
source "$TD/task.env"
mkdir -p "$(dirname "$D")"
( cd "$TD/pristine" && sha256sum --quiet -c "$TD/pristine.sha256" ) > "$D.pristine_check" 2>&1 || { mkdir -p "$D"; echo "pristine modified" > "$D/gate.txt"; exit 2; }
rm -rf "$D"; mkdir -p "$D/ws"
tar -C "$WS" --exclude=./dev_runs --exclude=.git --exclude=__pycache__ --exclude='*.pyc' -cf - . | tar -C "$D/ws" -xf -
python3 -I $H/kernel_gate.py --pristine "$TD/pristine" --candidate "$D/ws" --json "$D/gate.json" --stage "$D/stage" --patch "$D/changes.patch" > "$D/gate.txt" 2>&1
G=$?
[ $G -ne 0 ] && { echo "GATE REJECT (kernel_gate rc=$G)" >> "$D/gate.txt"; exit 1; }
python3 -I $H/lock_check.py --stage "$D/stage" --locked "$LOCKED" --json "$D/lock.json" >> "$D/gate.txt" 2>&1 || { echo "GATE REJECT (locked file changed)" >> "$D/gate.txt"; exit 1; }
# P2 rules on the diff (review F1/F4.3/F5): timing/exit hooks, harness paths, import machinery, process spawning, audit
# games on added lines; task-protected engine settings (GATE_PROTECT) on added AND removed lines
python3 -I $H/p2_gate_rules.py --patch "$D/changes.patch" ${GATE_PROTECT:+--protect "$GATE_PROTECT"} --json "$D/p2_rules.json" >> "$D/gate.txt" 2>&1 \
  || { echo "GATE REJECT (P2 rules)" >> "$D/gate.txt"; exit 1; }
echo "GATE PASS" >> "$D/gate.txt"; exit 0
