#!/bin/bash
# setup_prs.sh <pr:fixcommit:mergedate[:parentcommit[:override]]> ... : CPU/network only. For each ISO-Bench PR: parent commit via the GitHub API,
# per-commit vLLM wheels (wheels.vllm.ai, T5 fetch.py), one uv venv per commit with dependencies resolved as of merge
# date + 2 days (parent first; fix pinned to the parent's freeze), setuptools added, pyairports overridden + stubbed
# (old releases deleted from PyPI). Optional 5th field: one extra override "pkg==ver" for a pin whose release was deleted
# from PyPI (e.g. xgrammar==0.1.16 for vllm 0.8.2/0.8.3); only that package is exempt from the date cutoff.
# Records freeze diffs and which vllm package files differ.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
R=$ISO_ROOT; P2=$ISO_ROOT; S=$P2/screen
export UV_PYTHON_INSTALL_DIR=$R/.uv-python UV_PYTHON_BIN_DIR=$R/.uv-bin UV_CACHE_DIR=$P2/.uv-cache
UV=${UV:-uv}; SYSPY=$HOST_PY_BIN/python3; PY=$($UV python find 3.10)
mkdir -p $S/wheels $S/results $P2/venvs; cd $S
printf 'pyairports ; sys_platform == "never"\n' > results/uv_overrides.txt
stub() { mkdir -p "$1/pyairports"; printf '# P2 stub: PyPI no longer serves old pyairports releases\n' > "$1/pyairports/__init__.py"; printf 'AIRPORT_LIST = []\n' > "$1/pyairports/airports.py"; }
for spec in "$@"; do
  IFS=: read pr F D P OV <<< "$spec"
  X=$(date -d "$D + 2 days" +%F)
  echo "=== PR $pr fix=$F merged=$D exclude-newer=$X $(date +%T)"
  # parent: given in the spec (ISO-Bench data/mappings/all_95_baseline_commits.jsonl), else the GitHub API (rate-limited)
  [ -n "${P:-}" ] || P=$(curl -fsSL -H "User-Agent: p2-screen" https://api.github.com/repos/vllm-project/vllm/commits/$F | $SYSPY -I -c "import json,sys; d=json.load(sys.stdin); print(d['parents'][0]['sha'])") || { echo "PARENT LOOKUP FAILED $pr"; continue; }
  echo "parent=$P"; echo "$pr $P $F $X" >> results/commits.txt
  for role in parent fix; do c=$P; [ $role = fix ] && c=$F
    W=$(ls wheels/${pr}_$role/vllm-*.whl 2>/dev/null | head -1)
    [ -n "$W" ] || W=$($SYSPY -I fetch.py $pr $role $c wheels/${pr}_$role) || { echo "FETCH FAILED $pr $role"; continue 2; }
  done
  WP=$(ls wheels/${pr}_parent/vllm-*.whl); WF=$(ls wheels/${pr}_fix/vllm-*.whl)
  OVF=results/uv_overrides.txt; XNP=()
  if [ -n "${OV:-}" ]; then OVF=results/uv_overrides_$pr.txt; { cat results/uv_overrides.txt; echo "$OV"; } > $OVF; XNP=(--exclude-newer-package "${OV%%==*}=$(date +%F)"); echo "extra override: $OV"; fi
  for role in parent fix; do rm -rf $P2/venvs/${pr}_$role; $UV venv -q --python $PY $P2/venvs/${pr}_$role; done
  VIRTUAL_ENV=$P2/venvs/${pr}_parent $UV pip install -q --exclude-newer $X "${XNP[@]}" --override $OVF "$WP" setuptools || { echo "INSTALL FAILED ${pr}_parent"; continue; }
  VIRTUAL_ENV=$P2/venvs/${pr}_parent $UV pip freeze | grep -v -i "^vllm" > results/${pr}_constraints.txt
  VIRTUAL_ENV=$P2/venvs/${pr}_fix $UV pip install -q --exclude-newer $X "${XNP[@]}" --override $OVF -c results/${pr}_constraints.txt "$WF" setuptools || { echo "INSTALL FAILED ${pr}_fix"; continue; }
  for role in parent fix; do stub $P2/venvs/${pr}_$role/lib/python3.10/site-packages; VIRTUAL_ENV=$P2/venvs/${pr}_$role $UV pip freeze > results/${pr}_${role}_freeze.txt; done
  diff results/${pr}_parent_freeze.txt results/${pr}_fix_freeze.txt > results/${pr}_freeze.diff; echo "freeze diff lines: $(grep -c '^[<>]' results/${pr}_freeze.diff)"
  grep -i -E "^(torch|xformers|transformers|flashinfer-python|triton)==" results/${pr}_parent_freeze.txt | tr '\n' ' '; echo
  for role in parent fix; do (cd $P2/venvs/${pr}_$role/lib/python3.10/site-packages && find vllm -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > results/${pr}_${role}_vllm.sha256; done
  diff results/${pr}_parent_vllm.sha256 results/${pr}_fix_vllm.sha256 | grep '^[<>]' | awk '{print $3}' | sort -u > results/${pr}_vllm_pkg_files.txt
  echo "vllm files differing: $(tr '\n' ' ' < results/${pr}_vllm_pkg_files.txt)"
  for role in parent fix; do $P2/venvs/${pr}_$role/bin/python3 -I -c "import vllm, torch; print('$role import ok', vllm.__version__, torch.__version__, torch.version.cuda)" 2>&1 | tail -1; done
  echo "PR_DONE $pr"
done
echo SETUP_PRS_DONE
