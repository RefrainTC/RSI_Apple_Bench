#!/bin/bash
# setup_venvs.sh : build this package's venvs exactly as the authors had them (CPU + network only, no GPU).
#   1. uv CPython 3.10.20 into $ISO_ROOT/.uv-python (the venvs' base interpreter; stage_tmp.sh copies it to $STAGE)
#   2. one venv per role in $ISO_ROOT/venvs/<PR>_<role>, installed with --no-deps from env/<PR>_<role>.lock.txt
#      (every package pinned to the authors' version, every file sha256-checked) + the per-commit vLLM wheel (URL + sha256)
#   3. the authors' pyairports stub where they had one (old pyairports releases were deleted from PyPI; outlines imports it)
#   4. checks: `uv pip freeze` equals the authors' freeze (env/<PR>_<role>_freeze.txt, vllm line aside) and every file of
#      the installed vllm package hashes as in the authors' record (env/<PR>_<role>_vllm.sha256). Exit 1 on any difference.
# Settings: env/pkg.conf (PR, dates, wheels, override). Machine settings: env.sh / site.env (ISO_ROOT, UV).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"
. "$ISO_ROOT/env/pkg.conf"
export UV_PYTHON_INSTALL_DIR=$ISO_ROOT/.uv-python UV_PYTHON_BIN_DIR=$ISO_ROOT/.uv-bin UV_CACHE_DIR=$ISO_ROOT/.uv-cache UV_NO_CONFIG=1
LOG=$ISO_ROOT/setup/setup_venvs.log; exec > >(tee -a "$LOG") 2>&1
echo "=== $(date -Is) setup_venvs PR $PR (uv $($UV --version))"
$UV python install 3.10.20 || { echo "uv python install failed"; exit 1; }
PY=$ISO_ROOT/.uv-python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10
[ -x "$PY" ] || { echo "expected interpreter $PY not found (uv layout changed? use uv 0.11.14)"; exit 1; }
XNP=(); [ -n "${OVERRIDE_PKG:-}" ] && XNP=(--exclude-newer-package "$OVERRIDE_PKG=$OVERRIDE_UPLOADED_BY")
T=$(mktemp -d -p "${TMPDIR:-/tmp}" setupvenv.XXXXXX); trap 'rm -rf "$T"' EXIT
rc=0
for role in $ROLES; do
  V=$ISO_ROOT/venvs/${PR}_$role; t0=$(date +%s)
  url_var=WHEEL_URL_$role; sha_var=WHEEL_SHA256_$role
  { cat "$ISO_ROOT/env/${PR}_$role.lock.txt"; echo "vllm @ ${!url_var} --hash=sha256:${!sha_var}"; } > "$T/req_$role.txt"
  rm -rf "$V"; $UV venv -q --python "$PY" "$V" || { rc=1; continue; }
  VIRTUAL_ENV=$V $UV pip install -q --require-hashes --no-deps --index-url https://pypi.org/simple \
      --exclude-newer "$EXCLUDE_NEWER" "${XNP[@]}" -r "$T/req_$role.txt" || { echo "INSTALL FAILED $role"; rc=1; continue; }
  S=$V/lib/python3.10/site-packages
  case "${PYAIRPORTS_STUB:-p2}" in   # the authors' stub, byte for byte (pkg.conf says which one this PR's venvs had)
    p2) mkdir -p "$S/pyairports"; printf '# P2 stub: PyPI no longer serves old pyairports releases\n' > "$S/pyairports/__init__.py"; printf 'AIRPORT_LIST = []\n' > "$S/pyairports/airports.py" ;;
    t5) mkdir -p "$S/pyairports"; printf '# R4090 stub: PyPI no longer serves any pyairports release older than 2025-09; outlines only needs the name.\n' > "$S/pyairports/__init__.py"
        printf '# R4090 stub (see __init__.py)\nAIRPORT_LIST = []\n' > "$S/pyairports/airports.py" ;;
    none) ;;
  esac
  VIRTUAL_ENV=$V $UV pip freeze | grep -v -i -E '^vllm( |==|@)' > "$T/freeze_$role.txt"
  if diff <(grep -v -i -E '^vllm( |==|@)' "$ISO_ROOT/env/${PR}_${role}_freeze.txt") "$T/freeze_$role.txt" > "$T/fd_$role.txt"; then echo "$role freeze: identical to the authors' ($(wc -l < "$T/freeze_$role.txt") packages + vllm)"
  else echo "$role FREEZE DIFFERS:"; cat "$T/fd_$role.txt"; rc=1; fi
  (cd "$S" && find vllm -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$T/vllm_$role.sha256"
  if cmp -s "$T/vllm_$role.sha256" "$ISO_ROOT/env/${PR}_${role}_vllm.sha256"; then echo "$role vllm files: $(wc -l < "$T/vllm_$role.sha256") hashes identical to the authors' record"
  else echo "$role VLLM FILES DIFFER:"; diff "$T/vllm_$role.sha256" "$ISO_ROOT/env/${PR}_${role}_vllm.sha256" | head -20; rc=1; fi
  "$V/bin/python3" -I -c "import vllm, torch; print('$role import ok', vllm.__version__, torch.__version__, torch.version.cuda)" 2>&1 | tail -1
  echo "$role venv done in $(( $(date +%s) - t0 )) s, $(du -sh "$V" | cut -f1)"
done
[ "$(type -t post_setup)" = function ] && { post_setup || rc=1; }
echo "=== $(date -Is) setup_venvs rc=$rc"
exit $rc
