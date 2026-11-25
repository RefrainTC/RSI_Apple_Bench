#!/bin/bash
# setup_fix.sh : second pass after setup.sh.
#  (1) all venvs get setuptools (uv venvs ship without it; torch/triton import it at run time; pip-based images have it);
#  (2) 10733 / 7822: every old pyairports release has been deleted from PyPI (only a 2025-09 re-upload exists), so the
#      resolver cannot satisfy outlines>=0.0.43. pyairports is overridden out of the resolution and replaced by a local
#      empty stub (AIRPORT_LIST = []); it only feeds outlines' airport-code type, which these benchmarks never use.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
R=$ISO_ROOT; T5=$R/t5-isobench
export UV_PYTHON_INSTALL_DIR=$R/.uv-python UV_PYTHON_BIN_DIR=$R/.uv-bin UV_CACHE_DIR=$R/.uv-cache
UV=$HOME/.local/bin/uv
declare -A X; X[16436]=2025-04-23; X[7822]=2024-08-28; X[10733]=2024-11-30
cd $T5
printf 'pyairports ; sys_platform == "never"\n' > results/uv_overrides.txt
stub() {  # site-packages dir
  mkdir -p "$1/pyairports"
  printf '# R4090 stub: PyPI no longer serves any pyairports release older than 2025-09; outlines only needs the name.\n' > "$1/pyairports/__init__.py"
  printf '# R4090 stub (see __init__.py)\nAIRPORT_LIST = []\n' > "$1/pyairports/airports.py"
}
for pr in 16436 10733 7822; do
  echo "=== $pr $(date +%T)"
  WP=$(ls wheels/${pr}_parent/vllm-*.whl); WF=$(ls wheels/${pr}_fix/vllm-*.whl)
  if [ $pr = 16436 ]; then
    for role in parent fix; do VIRTUAL_ENV=$T5/venvs/${pr}_$role $UV pip install -q --exclude-newer ${X[$pr]} -c results/${pr}_constraints.txt setuptools; done
  else
    for role in parent fix; do rm -rf venvs/${pr}_$role; $UV venv -q --python $($UV python find 3.10) venvs/${pr}_$role; done
    VIRTUAL_ENV=$T5/venvs/${pr}_parent $UV pip install -q --exclude-newer ${X[$pr]} --override results/uv_overrides.txt "$WP" setuptools || { echo "INSTALL FAILED ${pr}_parent"; continue; }
    VIRTUAL_ENV=$T5/venvs/${pr}_parent $UV pip freeze | grep -v -i "^vllm" > results/${pr}_constraints.txt
    VIRTUAL_ENV=$T5/venvs/${pr}_fix $UV pip install -q --exclude-newer ${X[$pr]} --override results/uv_overrides.txt -c results/${pr}_constraints.txt "$WF" setuptools || { echo "INSTALL FAILED ${pr}_fix"; continue; }
    for role in parent fix; do stub $T5/venvs/${pr}_$role/lib/python3.10/site-packages; done
  fi
  for role in parent fix; do VIRTUAL_ENV=$T5/venvs/${pr}_$role $UV pip freeze > results/${pr}_${role}_freeze.txt; done
  diff results/${pr}_parent_freeze.txt results/${pr}_fix_freeze.txt > results/${pr}_freeze.diff; echo "freeze diff lines: $(grep -c '^[<>]' results/${pr}_freeze.diff)"
  grep -i -E "^(torch|xformers|transformers|vllm-flash-attn|flashinfer|triton|setuptools|outlines)==" results/${pr}_parent_freeze.txt | tr '\n' ' '; echo
  for role in parent fix; do (cd venvs/${pr}_$role/lib/python3.10/site-packages && find vllm -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > results/${pr}_${role}_vllm.sha256; done
  diff results/${pr}_parent_vllm.sha256 results/${pr}_fix_vllm.sha256 | grep '^[<>]' | awk '{print $1, $3}' | sort -k2 > results/${pr}_vllm_pkg.diff
  echo "vllm package files differing: $(awk '{print $2}' results/${pr}_vllm_pkg.diff | sort -u | tr '\n' ' ')"
  for role in parent fix; do venvs/${pr}_$role/bin/python3 -I -c "import vllm, torch, setuptools; print('$role import ok', vllm.__version__, torch.__version__, torch.version.cuda)" 2>&1 | tail -1; done
done
echo SETUP_FIX_DONE
