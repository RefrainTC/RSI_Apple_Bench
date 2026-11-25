#!/bin/bash
# stage_n1.sh : stage the #12380 inputs to THIS pod's $P2T (run on n1). Uses the shared stage_tmp.sh unchanged.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
. $ISO_ROOT/ops_iso12380/common.sh
t0=$(date +%s)
taskset -c 0-7,32-39 nice -n 10 bash $P2/scripts/stage_tmp.sh py venv:$P2/venvs/12380_parent venv:$P2/venvs/12380_fix \
  hf:RedHatAI/Meta-Llama-3-8B-Instruct-FP8 > $LOGD/stage_n1.log 2>&1
rc=$?; mkdir -p $P2T/tmp && chmod 700 $P2T/tmp
echo "$(date '+%F %T') STAGE rc=$rc secs=$(( $(date +%s) - t0 ))" >> $LOGD/stage_n1.log
