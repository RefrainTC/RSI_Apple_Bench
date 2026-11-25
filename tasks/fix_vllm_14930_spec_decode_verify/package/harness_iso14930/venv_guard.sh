#!/bin/bash
# venv_guard.sh <pre|post> <master venv> <run venv> <record.json> : per-run venv restore + verification (review F4).
# The job never runs from the staged master: it runs from a per-GPU copy (<run venv>, $P2T/run/g<gpu>/<name>).
# Right after a copy is made, its listing (path, size, symlink target, inode, ctime) is saved as the baseline. ctime cannot
# be set by a user process and changes on every write, chmod or rename, so any change to any file - even same size with
# a forged mtime - shows up in the listing.
#   pre : copy if missing; if the copy's listing differs from its baseline, replace it by a fresh copy (and say so).
#   post: compare with the baseline; a difference means the run changed the venv -> record changed_by_run (the harness
#         voids the run) and restore a fresh copy for the next run.
# The master itself is checked against the off-pod sha256 manifest by integrity.sh at the start and end of every
# calibration / scoring job.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
PH=$1; M=$(readlink -f "$2"); RV=$3; REC=$4
lst() { (cd "$1" && find . \( -type f -o -type l \) -printf '%P %s %l %i %C@\n' | LC_ALL=C sort); }
fresh() {
  local t0=$(date +%s.%N); rm -rf "$RV" "$RV.baseline"; mkdir -p "$(dirname "$RV")"
  cp -a "$M" "$RV" || { echo "copy failed"; return 1; }
  lst "$RV" > "$RV.baseline"
  awk -v a=$t0 -v b=$(date +%s.%N) 'BEGIN{printf "%.1f", b-a}'
}
state=ok; copied=""; ndiff=0
if [ "$PH" = pre ]; then
  if [ ! -d "$RV" ] || [ ! -f "$RV.baseline" ]; then copied=$(fresh) || exit 1; state=fresh_copy
  else
    ndiff=$(diff <(lst "$RV") "$RV.baseline" | grep -c '^[<>]')
    if [ "$ndiff" != 0 ]; then diff <(lst "$RV") "$RV.baseline" | head -50 > "${REC%.json}.pre_diff.txt"; copied=$(fresh) || exit 1; state=changed_before_run_restored; fi
  fi
else
  ndiff=$(diff <(lst "$RV") "$RV.baseline" | grep -c '^[<>]')
  if [ "$ndiff" != 0 ]; then diff <(lst "$RV") "$RV.baseline" | head -50 > "${REC%.json}.post_diff.txt"; state=changed_by_run; copied=$(fresh) || exit 1; fi
fi
python3 -I - "$REC" "$PH" "$state" "$ndiff" "${copied:-}" "$M" "$RV" <<'PY'
import json, os, sys
f, ph, st, nd, cp, m, rv = sys.argv[1:8]
d = json.load(open(f)) if os.path.exists(f) else {}
d[ph] = {"state": st, "diff_lines": int(nd), "copy_s": float(cp) if cp else None, "master": m, "run_venv": rv}
json.dump(d, open(f, "w"))
PY
echo "$RV"
