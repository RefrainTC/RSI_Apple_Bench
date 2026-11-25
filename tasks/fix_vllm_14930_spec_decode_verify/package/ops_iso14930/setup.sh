#!/bin/bash
# setup.sh : create harness_iso14930 (copy of harness/ with its own paths + score.sh operator-stage mode), install the
# task dir, build workspace/pristine, oracle stage. CPU only.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -euo pipefail
P2=$ISO_ROOT; HN=$P2/harness_iso14930; IN=$P2/ops_iso14930/incoming
mkdir -p $P2T/tmp && chmod 700 $P2T/tmp
if [ ! -d "$HN" ]; then
  cp -a $P2/harness "$HN"; rm -rf "$HN/__pycache__"
  grep -rl --include='*.sh' --include='*.py' -e '$ISO_ROOT/harness' -e '\$P2/harness' "$HN" | while read -r f; do
    sed -i -e 's#$ISO_ROOT/harness\([^_]\|$\)#$ISO_ROOT/harness_iso14930\1#g' -e 's#\$P2/harness\([^_]\|$\)#$P2/harness_iso14930\1#g' "$f"; done
  sed -i 's#find harness -type f#find harness_iso14930 -type f#' "$HN/integrity.sh"
  cp "$IN/score.sh" "$HN/score.sh"
  printf 'harness_iso14930 = copy of p2-port/harness made %s\n- every path to harness/ points to harness_iso14930/\n- integrity.sh hashes harness_iso14930/\n- score.sh: STAGE:<dir> mode (operator only) scores a prepared stage with the gate bypassed (human-written PR Triton, canaries on it)\n' "$(date -Is)" > "$HN/CHANGES_iso14930.txt"
fi
grep -rn 'p2-port/harness[^_]' "$HN" --include='*.sh' --include='*.py' | grep -v '^.*#' | head || true
TD=$P2/tasks/iso14930; mkdir -p $TD
cp "$IN"/task/* $TD/; chmod +x $TD/run.sh
ls -la $TD
[ -f $P2T/.staged/venv_14930_parent ] && [ -f $P2T/.staged/venv_14930_fix ] && echo staged-ok
ls $P2T/hf/hub | grep -i qwen2.5-1.5b
if [ ! -d $TD/workspace ]; then bash $HN/build_workspace.sh $TD; fi
SITEP=$($P2T/venvs/14930_parent/bin/python3 -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
SITEF=$($P2T/venvs/14930_fix/bin/python3 -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
mkdir -p $TD/calib; python3 -I $HN/mk_oracle_stage.py "$SITEP" "$SITEF" vllm $TD/calib/oracle_stage
echo SETUP DONE
