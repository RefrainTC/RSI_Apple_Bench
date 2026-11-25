#!/bin/bash
# remote_dev_entry.sh <task> <snapshot.tgz> <dev_id> [--baseline] [--correct] [VAR=VAL ...]
# Trial mode with the solver OFF the pod: the solver's local dev_run.sh ships a tar.gz of its workspace here (repo/ and
# NOTES.md are used, see unpack_snapshot.py). This entry point
#   1 unpacks it into $TD/remote_in/<dev_id>/ws (untrusted: regular files only, no links / absolute / '..' paths),
#   2 runs exactly what dev_run_impl.sh does on that workspace: snapshot + gate (kernel gate, locked files, P2 diff
#     rules) as the pod-side second line of defence, the candidate run, optional baseline and correctness runs,
#   3 writes the agent-visible results to $TD/remote_out/<dev_id>/ (status.txt, summary.json, gate.txt, log tails),
#     which the local wrapper copies back. Nothing else under $TD is meant for the solver.
# Exit: 0 done, 1 gate reject or invalid snapshot, 2 bad arguments. The GPU is the task's (task.env / $TD/trial_gpu).
# Terminal marker for a detached caller: $OUT/exit_code (written last). status.txt goes unpacking -> queued -> running ->
# done | gate_reject | invalid_snapshot.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; H=$P2/harness_iso13365
[ $# -ge 3 ] || { echo "usage: remote_dev_entry.sh <task> <snapshot.tgz> <dev_id> [--baseline] [--correct] [VAR=VAL ...]"; exit 2; }
T=$1; TGZ=$(readlink -f "$2"); ID=$3; shift 3
[[ "$T" =~ ^[A-Za-z0-9_]+$ ]] && [ -f "$P2/tasks/$T/task.env" ] || { echo "unknown task $T"; exit 2; }
[[ "$ID" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || { echo "bad dev id (A-Za-z0-9._-, at most 64)"; exit 2; }
TD=$P2/tasks/$T; IN=$TD/remote_in/$ID; OUT=$TD/remote_out/$ID
[ -e "$IN" ] || [ -e "$OUT" ] || [ -e "$TD/staging/$ID" ] && { echo "dev id $ID already used"; exit 2; }
mkdir -p "$IN" "$OUT"; echo "unpacking" > "$OUT/status.txt"
python3 -I $H/unpack_snapshot.py "$TGZ" "$TD/pristine" "$IN/ws" "$OUT/unpack.json" > "$IN/unpack.log" 2>&1 || {
  echo "invalid_snapshot" > "$OUT/status.txt"; cat "$OUT/unpack.json" 2>/dev/null; echo 1 > "$OUT/exit_code"; exit 1; }
sha256sum "$TGZ" | cut -c1-64 > "$IN/snapshot.sha256"; cp "$TGZ" "$IN/snapshot.tgz"
bash $H/dev_run_impl.sh "$TD" --ws "$IN/ws" --out "$OUT" --id "$ID" "$@"; RC=$?
printf '{"dev_id":"%s","snapshot_sha256":"%s","args":"%s","rc":%d,"ts":"%s"}\n' "$ID" "$(cat $IN/snapshot.sha256)" "$*" $RC "$(date -Is)" >> $TD/logs/remote_dev.jsonl
echo $RC > "$OUT/exit_code"
exit $RC
