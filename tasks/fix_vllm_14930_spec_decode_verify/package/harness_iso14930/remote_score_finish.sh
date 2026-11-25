#!/bin/bash
# remote_score_finish.sh <task> <label> (vault key on stdin) : analysis phase of a score whose run phase ran while the
# task was sealed (remote_score_entry.sh / score.sh stop with results/<label>/phase1.json). Refuses if any process of the
# score's jobs is still alive or the task GPU has a compute process; then restores only the reference files
# (vault.sh open-refs), runs score_phase2.sh (integrity of the restored refs, correctness, summarize, verdict) and deletes
# the reference files again whatever happens.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; H=$P2/harness_iso14930
[ $# -eq 2 ] || { echo "usage: remote_score_finish.sh <task> <label>  (key on stdin)"; exit 2; }
T=$1; L=$2; [[ "$T" =~ ^[A-Za-z0-9_]+$ ]] && [[ "$L" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || { echo "bad args"; exit 2; }
TD=$P2/tasks/$T; R=$TD/results/$L; source "$TD/task.env"
[ -f "$R/phase1.json" ] || { echo "no run phase for $L"; exit 2; }
IFS= read -r KEY || true
alive=""
for d in /proc/[0-9]*; do
  e=$(tr '\0' '\n' < $d/environ 2>/dev/null | grep -m1 -E "^R4_TAG=p2s_${TASK}_${L}_|^P2_AUDIT_OUT=$R/")
  [ -n "$e" ] && alive="$alive ${d#/proc/}"
done
apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ -n "$alive" ] || [ "$apps" != 0 ]; then
  echo "{\"verdict\":\"refused_alive\",\"pids\":\"$alive\",\"gpu_apps\":$apps}" > "$R/finish_refused.json"
  echo "refusing to restore references: score processes alive [$alive] / GPU$GPU compute apps $apps"; exit 3
fi
trap 'bash $H/vault.sh close-refs "$TD" > /dev/null' EXIT
printf '%s\n' "$KEY" | bash $H/vault.sh open-refs "$TD" || exit 1
unset KEY
bash $H/score_phase2.sh "$TD" "$L"
