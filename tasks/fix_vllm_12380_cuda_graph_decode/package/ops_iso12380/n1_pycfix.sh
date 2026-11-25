#!/bin/bash
# n1_pycfix.sh : make n1's staged 12380 venv masters byte-identical to main's (main's masters hold 368 __pycache__ .pyc
# files written by the batch-2 screen runs; the NFS source does not), so the shared integrity manifest verifies on n1.
# Waits for the n1 smoke to end, adds the files, drops n1's per-run venv copies (re-copied from the master next run).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; X=$P2/ops_iso12380/xfer; L=$P2/ops_iso12380/logs/n1_pycfix.log
until grep -q "SMOKE DONE" $P2/smoke/iso12380_n1/smoke.log 2>/dev/null; do sleep 20; done
for v in 12380_parent 12380_fix; do
  tar -C $P2T/venvs/$v -xf $X/$v.extra.tar
  n=$(cd $P2T/venvs/$v && find . -type f ! -name pyvenv.cfg ! -path "*/__pycache__/*.pyc.*" | wc -l)
  rm -rf $P2T/run/g0/$v $P2T/run/g0/$v.baseline
  echo "$(date '+%F %T') $v files=$n (main 28744)" >> $L
done
