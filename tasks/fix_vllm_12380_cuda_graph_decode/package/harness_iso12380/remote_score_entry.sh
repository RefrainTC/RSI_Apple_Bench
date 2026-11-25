#!/bin/bash
# remote_score_entry.sh <task> <snapshot.tgz> <label> [--transfer] : final scoring of a solver's snapshot (operator only;
# trial mode with the solver off the pod). Unpacks like remote_dev_entry.sh into $TD/remote_in/score_<label>/ws, then
# runs harness/score.sh on it. Results: $TD/results/<label>/ (summary.json, summary.md, gate.txt, changes.patch, runs/).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; H=$P2/harness_iso12380
[ $# -ge 3 ] || { echo "usage: remote_score_entry.sh <task> <snapshot.tgz> <label> [--transfer]"; exit 2; }
T=$1; TGZ=$(readlink -f "$2"); L=$3; TR=${4:-}
[[ "$T" =~ ^[A-Za-z0-9_]+$ ]] && [ -f "$P2/tasks/$T/task.env" ] || { echo "unknown task $T"; exit 2; }
[[ "$L" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || { echo "bad label"; exit 2; }
[ -z "$TR" ] || [ "$TR" = --transfer ] || { echo "unknown option $TR"; exit 2; }
TD=$P2/tasks/$T; IN=$TD/remote_in/score_$L
[ -e "$IN" ] || [ -e "$TD/results/$L" ] && { echo "label $L already used"; exit 2; }
mkdir -p "$IN"
python3 -I $H/unpack_snapshot.py "$TGZ" "$TD/pristine" "$IN/ws" "$IN/unpack.json" > "$IN/unpack.log" 2>&1 || {
  mkdir -p "$TD/results/$L"; echo '{"verdict":"invalid_snapshot"}' > "$TD/results/$L/summary.json"; cat "$IN/unpack.json"; exit 1; }
sha256sum "$TGZ" | cut -c1-64 > "$IN/snapshot.sha256"; cp "$TGZ" "$IN/snapshot.tgz"
bash $H/score.sh "$TD" "$L" "$IN/ws" $TR; RC=$?
cp "$IN/snapshot.sha256" "$TD/results/$L/snapshot.sha256" 2>/dev/null
exit $RC
