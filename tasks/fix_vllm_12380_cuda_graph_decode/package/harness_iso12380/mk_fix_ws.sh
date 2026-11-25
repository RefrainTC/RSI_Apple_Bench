#!/bin/bash
# mk_fix_ws.sh <taskdir> <out workspace> : a test workspace = pristine workspace + the human fix (calib/oracle_stage) applied
# to repo/, used for the end-to-end harness check (score.sh should give ~REF and pass correctness).
set -euo pipefail
TD=$(readlink -f "$1"); O=$2
rm -rf "$O"; cp -a "$TD/workspace" "$O"; rm -rf "$O/dev_runs"
( cd "$TD/calib/oracle_stage/repo_overlay" && find . -type f -print0 | xargs -0 -I{} cp {} "$O/repo/{}" )
[ -s "$TD/calib/oracle_stage/repo_deleted.txt" ] && while read -r r; do rm -f "$O/repo/$r"; done < "$TD/calib/oracle_stage/repo_deleted.txt"
echo "fix workspace $O"
