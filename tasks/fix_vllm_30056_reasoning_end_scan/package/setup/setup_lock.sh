#!/bin/bash
# setup_lock.sh : build the parent and fix venvs exactly as the authors had them (CPU and network only, ~5-10 min).
#   1. uv-managed CPython 3.10.20 under $ISO_ROOT/.uv-python
#   2. per role (parent, fix): uv venv; all dependencies from the hashed lock env/deps.lock + the role's per-commit vLLM
#      wheel from wheels.vllm.ai (URL and sha256 in env/build.env), all --no-deps --require-hashes
#   3. bytecode pre-compiled (compileall), so that runs do not write into the venv (the harness checks that)
#   4. checks: uv pip freeze equals the authors' env/<role>_freeze.txt (apart from the local path of the vllm line);
#      every installed vllm/ file hashes as in env/<role>_vllm.sha256; import check
# Output: $ISO_ROOT/venvs/<pr>_{parent,fix}, records in $ISO_ROOT/setup_out/. Ends with SETUP_LOCK_OK <pr> only if every
# check passes.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
E=$ISO_ROOT/env; S=$ISO_ROOT/setup_out; mkdir -p $S $ISO_ROOT/venvs
export UV_PYTHON_INSTALL_DIR=$ISO_ROOT/.uv-python UV_PYTHON_BIN_DIR=$ISO_ROOT/.uv-bin UV_CACHE_DIR=${UV_CACHE_DIR:-$ISO_ROOT/.uv-cache}
export UV_PYTHON_PREFERENCE=only-managed UV_HTTP_TIMEOUT=300 UV_NO_CONFIG=1
fail() { echo "SETUP FAILED: $*"; exit 1; }
t0=$(date +%s); echo "== setup_lock $PR fix=$FIX parent=$PARENT cutoff=$XNEWER  $("$UV" --version)"
"$UV" python install 3.10.20 || fail "uv python install 3.10.20"
PY=$("$UV" python find 3.10.20) || fail "python 3.10.20 not found"; echo "interpreter $PY"
T=$(mktemp -d -p "${TMPDIR:-/tmp}" c5ysetup.XXXXXX); trap 'rm -rf "$T"' EXIT
ok=1
for role in parent fix; do
  V=$ISO_ROOT/venvs/${PR}_$role; u=WHEEL_URL_$role; h=WHEEL_SHA256_$role
  { cat $E/deps.lock; echo "vllm @ ${!u} --hash=sha256:${!h}"; } > $T/req_$role.txt
  rm -rf "$V"; "$UV" venv -q --python "$PY" "$V" || fail "venv $role"
  for k in 1 2 3; do VIRTUAL_ENV=$V "$UV" pip install -q --no-deps --require-hashes --index-url https://pypi.org/simple -r $T/req_$role.txt && break
    [ $k = 3 ] && fail "install $role"; echo "install attempt $k failed; retrying"; sleep 20; done
  "$V/bin/python3" -I -m compileall -q -j 8 "$V/lib/python3.10/site-packages" > $S/${role}_compileall.log 2>&1 || true
  echo "$role: bytecode compiled ($(find "$V" -name '*.pyc' | wc -l) .pyc; failures for py2-only test files are expected)"
  VIRTUAL_ENV=$V "$UV" pip freeze > $S/${role}_freeze.txt
  norm() { sed -E 's#^vllm @ .*#vllm @ <wheel>#' "$1"; }
  if diff <(norm $E/${role}_freeze.txt) <(norm $S/${role}_freeze.txt) > $S/${role}_freeze_vs_authors.diff; then echo "$role: freeze equals the authors' ($(wc -l < $S/${role}_freeze.txt) packages)"
  else echo "$role: FREEZE DIFFERS from the authors' (see $S/${role}_freeze_vs_authors.diff)"; ok=0; fi
  (cd $V/lib/python3.10/site-packages && find vllm -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $S/${role}_vllm.sha256
  if cmp -s $S/${role}_vllm.sha256 $E/${role}_vllm.sha256; then echo "$role: $(wc -l < $S/${role}_vllm.sha256) vllm files hash exactly as the authors' build"
  else echo "$role: VLLM FILES DIFFER from the authors' build"; ok=0; fi
  r=$(cd $T && $V/bin/python3 -I -c "import vllm, torch; print('$role import ok', vllm.__version__, torch.__version__, torch.version.cuda)" 2>&1 | tail -1); echo "$r"
  case "$r" in *"import ok"*) ;; *) echo "$role: import check failed"; ok=0;; esac
done
diff $S/parent_vllm.sha256 $S/fix_vllm.sha256 | grep '^[<>]' | awk '{print $3}' | sort -u > $S/files_differing.txt
echo "files differing parent vs fix: $(tr '\n' ' ' < $S/files_differing.txt)"
echo "venv sizes: $(du -sh $ISO_ROOT/venvs/${PR}_parent $ISO_ROOT/venvs/${PR}_fix | awk '{print $1}' | tr '\n' ' ')  time $(( $(date +%s) - t0 )) s"
[ $ok = 1 ] && { echo "SETUP_LOCK_OK $PR"; exit 0; } || fail "checks above"
