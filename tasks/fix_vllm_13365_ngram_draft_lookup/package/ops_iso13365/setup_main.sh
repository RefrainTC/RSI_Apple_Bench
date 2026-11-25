#!/bin/bash
# setup_main.sh : one-time setup of iso13365 on the main pod (CPU / disk only, no GPU). Idempotent where possible.
#  1. NAS venv 13365_base = 13365_parent + numba 0.60.0 / llvmlite 0.43.0 copied from 13365_fix (the fix's own pins)
#  2. stage it to $P2T/venvs (shared stage_tmp.sh, unchanged), import check
#  3. workspace + pristine (build_workspace.sh of the harness copy), fix workspace, gate check of the fix workspace
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; V=$P2/venvs; H=$P2/harness_iso13365; TD=$P2/tasks/iso13365; OPS=$P2/ops_iso13365
mkdir -p $P2T/tmp && chmod 700 $P2T/tmp
SP=lib/python3.10/site-packages
if [ ! -f $V/13365_base/.iso13365_built ]; then
  echo "$(date -u '+%F %T') build NAS venv 13365_base"
  rm -rf $V/13365_base.partial && cp -a $V/13365_parent $V/13365_base.partial || exit 1
  for d in numba numba-0.60.0.dist-info llvmlite llvmlite-0.43.0.dist-info llvmlite.libs; do
    cp -a $V/13365_fix/$SP/$d $V/13365_base.partial/$SP/ || exit 1; done
  rm -rf $V/13365_base && mv $V/13365_base.partial $V/13365_base && date -Is > $V/13365_base/.iso13365_built
  echo "$(date -u '+%F %T') built: $(find $V/13365_base -type f | wc -l) files"
fi
nice -n 10 taskset -c 8-15,40-47 bash $P2/scripts/stage_tmp.sh venv:$V/13365_base || exit 1
$P2T/venvs/13365_base/bin/python3 -I -c "import numba, llvmlite, numpy, vllm; print('base import ok', numba.__version__, llvmlite.__version__, numpy.__version__, vllm.__version__)" || exit 1
$P2T/venvs/13365_base/bin/python3 -I -c "import vllm.v1.spec_decode.ngram_proposer as m, inspect; print('parent proposer has numba:', 'numba' in inspect.getsource(m))"
# workspace / pristine / fix workspace
[ -d $TD/workspace ] || bash $H/build_workspace.sh $TD || exit 1
echo "$(date -u '+%F %T') setup_main done"
