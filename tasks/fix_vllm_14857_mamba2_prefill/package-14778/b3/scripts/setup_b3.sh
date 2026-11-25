#!/bin/bash
# setup_b3.sh <pr:fixcommit:mergedate:parentcommit[:override]> ... : batch-3 copy of p2-port/screen/setup_prs.sh (CPU and
# network only). Same recipe (per-commit wheels from wheels.vllm.ai via screen/fetch.py, uv venv per commit, deps as of
# merge date + 2 days, fix pinned to the parent's freeze, pyairports stub, optional one override exempt from the date
# cutoff); writes under b3/ only: wheels b3/setup/wheels, venvs b3/venvs, records b3/setup/, markers done_<pr>/failed_<pr>.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../../env.sh"; R=$ISO_ROOT; P2=$ISO_ROOT; B3=$P2/b3; S=$B3/setup
export UV_PYTHON_INSTALL_DIR=$R/.uv-python UV_PYTHON_BIN_DIR=$R/.uv-bin UV_CACHE_DIR=$P2/.uv-cache UV_PYTHON_PREFERENCE=only-managed
PY=$($UV python find 3.10.20)
mkdir -p $S/wheels $B3/venvs; cd $S
printf 'pyairports ; sys_platform == "never"\n' > uv_overrides.txt
stub() { mkdir -p "$1/pyairports"; printf '# P2 stub: PyPI no longer serves old pyairports releases\n' > "$1/pyairports/__init__.py"; printf 'AIRPORT_LIST = []\n' > "$1/pyairports/airports.py"; }
fail() { echo "FAILED $1: $2"; echo "$2" > $S/failed_$1; }
for spec in "$@"; do
  IFS=: read pr F D P OV <<< "$spec"
  X=$(date -d "$D + 2 days" +%F); rm -f $S/done_$pr $S/failed_$pr
  echo "=== PR $pr fix=$F parent=$P merged=$D exclude-newer=$X $(date -u +%T)"
  for role in parent fix; do c=$P; [ $role = fix ] && c=$F
    W=$(ls wheels/${pr}_$role/vllm-*.whl 2>/dev/null | head -1)
    [ -n "$W" ] || W=$($SYSPY -I $P2/screen/fetch.py $pr $role $c wheels/${pr}_$role) || { fail $pr "fetch $role"; continue 2; }
  done
  WP=$(ls wheels/${pr}_parent/vllm-*.whl); WF=$(ls wheels/${pr}_fix/vllm-*.whl)
  OVF=uv_overrides.txt; XNP=()
  if [ -n "${OV:-}" ]; then OVF=uv_overrides_$pr.txt; { cat uv_overrides.txt; echo "$OV"; } > $OVF; XNP=(--exclude-newer-package "${OV%%==*}=$(date +%F)"); echo "extra override: $OV"; fi
  for role in parent fix; do rm -rf $B3/venvs/${pr}_$role; $UV venv -q --python $PY $B3/venvs/${pr}_$role; done
  VIRTUAL_ENV=$B3/venvs/${pr}_parent $UV pip install -q --exclude-newer $X "${XNP[@]}" --override $OVF "$WP" setuptools || { fail $pr "install parent"; continue; }
  VIRTUAL_ENV=$B3/venvs/${pr}_parent $UV pip freeze | grep -v -i "^vllm" > ${pr}_constraints.txt
  VIRTUAL_ENV=$B3/venvs/${pr}_fix $UV pip install -q --exclude-newer $X "${XNP[@]}" --override $OVF -c ${pr}_constraints.txt "$WF" setuptools || { fail $pr "install fix"; continue; }
  for role in parent fix; do stub $B3/venvs/${pr}_$role/lib/python3.10/site-packages; VIRTUAL_ENV=$B3/venvs/${pr}_$role $UV pip freeze > ${pr}_${role}_freeze.txt; done
  diff ${pr}_parent_freeze.txt ${pr}_fix_freeze.txt > ${pr}_freeze.diff; echo "freeze diff lines: $(grep -c '^[<>]' ${pr}_freeze.diff)"
  grep -i -E "^(torch|xformers|transformers|flashinfer-python|triton|mamba-ssm)==" ${pr}_parent_freeze.txt | tr '\n' ' '; echo
  for role in parent fix; do (cd $B3/venvs/${pr}_$role/lib/python3.10/site-packages && find vllm -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > ${pr}_${role}_vllm.sha256; done
  diff ${pr}_parent_vllm.sha256 ${pr}_fix_vllm.sha256 | grep '^[<>]' | awk '{print $3}' | sort -u > ${pr}_vllm_pkg_files.txt
  echo "vllm files differing: $(tr '\n' ' ' < ${pr}_vllm_pkg_files.txt)"
  ok=1; for role in parent fix; do $B3/venvs/${pr}_$role/bin/python3 -I -c "import vllm, torch; print('$role import ok', vllm.__version__, torch.__version__, torch.version.cuda)" 2>&1 | tail -1 | tee /dev/stderr | grep -q "import ok" || ok=0; done
  [ $ok = 1 ] && { date -u +%FT%T > $S/done_$pr; echo "PR_DONE $pr"; } || fail $pr "import check"
done
echo SETUP_B3_DONE
