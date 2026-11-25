#!/bin/bash
# setup_lock.sh : build the parent and fix venvs exactly as the authors had them (CPU and network only, ~5-10 min).
#   1. uv-managed CPython 3.10.20 under $ISO_ROOT/.uv-python
#   2. per role (parent, fix): uv venv; all dependencies from the hashed lock env/deps.lock (--no-deps --require-hashes),
#      then $REMOVE_AFTER_LOCK is uninstalled if env/build.env names one (here: none);
#      SGLang = python/ of the commit's source tarball (codeload.github.com), every file checked against the authors'
#      env/<role>_src.sha256, then built by the backend its pyproject.toml declares, --no-deps
#   3. checks: uv pip freeze equals the authors' env/<role>_freeze.txt (apart from the local path of the sglang line);
#      every installed sglang/ file hashes as in env/<role>_sgl.sha256; import check
# Output: $ISO_ROOT/venvs/<pr>_{parent,fix}, records in $ISO_ROOT/setup_out/. Ends with SETUP_LOCK_OK <pr> only if every
# check passes.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
E=$ISO_ROOT/env; S=$ISO_ROOT/setup_out; mkdir -p $S $ISO_ROOT/venvs $ISO_ROOT/src
export UV_PYTHON_INSTALL_DIR=$ISO_ROOT/.uv-python UV_PYTHON_BIN_DIR=$ISO_ROOT/.uv-bin UV_CACHE_DIR=${UV_CACHE_DIR:-$ISO_ROOT/.uv-cache}
export UV_PYTHON_PREFERENCE=only-managed UV_HTTP_TIMEOUT=300 SGLANG_BUILD_RUST_EXTS=none
fail() { echo "SETUP FAILED: $*"; exit 1; }
t0=$(date +%s); echo "== setup_lock $PR fix=$FIX parent=$PARENT cutoff=$XNEWER  $("$UV" --version)"
"$UV" python install 3.10.20 || fail "uv python install 3.10.20"
PY=$("$UV" python find 3.10.20) || fail "python 3.10.20 not found"; echo "interpreter $PY"
ok=1
for role in parent fix; do
  c=$PARENT; [ $role = fix ] && c=$FIX; V=$ISO_ROOT/venvs/${PR}_$role
  rm -rf "$V"; "$UV" venv -q --python "$PY" "$V" || fail "venv $role"
  # the lock holds PyPI wheels only (torch 2.9.1 = the CUDA 12.8 build, sgl-kernel 0.3.19), all hash-pinned
  for k in 1 2 3; do VIRTUAL_ENV=$V "$UV" pip install -q --no-deps --require-hashes -r $E/deps.lock && break
    [ $k = 3 ] && fail "dependencies $role"; echo "dependency install attempt $k failed; retrying"; sleep 20; done
  u=URL_$role; d=$ISO_ROOT/src/sgl_$c
  if [ ! -f $d/.ok ]; then rm -rf $d; mkdir -p $d
    for k in 1 2 3 4 5; do curl -fsSL --retry 3 -o $d.tgz "${!u}" && break; [ $k = 5 ] && fail "download source $role"; sleep 20; done
    tar -C $d --strip-components=1 --wildcards -xzf $d.tgz "sglang-*/python/*" || fail "untar $role"; rm -f $d.tgz
    (cd $d && sha256sum --quiet -c $E/${role}_src.sha256) || fail "source files differ from the authors' ($role)"
    n1=$(wc -l < $E/${role}_src.sha256); n2=$(cd $d && find python -type f | wc -l)
    [ "$n1" = "$n2" ] || fail "source file count $n2 != authors' $n1 ($role)"
    echo "$role: all $n1 source files of python/ match the authors' hashes"
    # as the authors did (batch 4): the tree declares a Rust extension (sglang.srt.grpc._core) statically and no cargo is
    # assumed; drop that pyproject block (the offline Engine never imports the gRPC core); identical in both roles
    awk 'BEGIN{skip=0} /^\[\[tool.setuptools-rust.ext-modules\]\]/{skip=1; next} skip && /^\[/{skip=0} !skip{print}' $d/python/pyproject.toml > $d/python/pyproject.toml.new && mv $d/python/pyproject.toml.new $d/python/pyproject.toml
    touch $d/.ok; fi
  VIRTUAL_ENV=$V "$UV" pip install -q --no-deps --exclude-newer $XNEWER "$d/python" || fail "install sglang $role"
  [ -z "${REMOVE_AFTER_LOCK:-}" ] || VIRTUAL_ENV=$V "$UV" pip uninstall -q $REMOVE_AFTER_LOCK || fail "uninstall $REMOVE_AFTER_LOCK $role"
  VIRTUAL_ENV=$V "$UV" pip freeze > $S/${role}_freeze.txt
  norm() { sed -E 's#^sglang @ .*#sglang @ <local>#' "$1" | { if [ -n "${REMOVE_AFTER_LOCK:-}" ]; then grep -v -E "^${REMOVE_AFTER_LOCK}=="; else cat; fi; }; }
  if diff <(norm $E/${role}_freeze.txt) <(norm $S/${role}_freeze.txt) > $S/${role}_freeze_vs_authors.diff; then echo "$role: freeze equals the authors' ($(wc -l < $S/${role}_freeze.txt) packages)"
  else echo "$role: FREEZE DIFFERS from the authors' (see $S/${role}_freeze_vs_authors.diff)"; ok=0; fi
  (cd $V/lib/python3.10/site-packages && find sglang -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $S/${role}_sgl.sha256
  if cmp -s $S/${role}_sgl.sha256 $E/${role}_sgl.sha256; then echo "$role: $(wc -l < $S/${role}_sgl.sha256) sglang files hash exactly as the authors' build"
  elif cmp -s <(grep -v ' sglang/_version.py$' $S/${role}_sgl.sha256) <(grep -v ' sglang/_version.py$' $E/${role}_sgl.sha256) \
       && [ "$(cd $S && $V/bin/python3 -I -c 'import sglang._version as v; print(v.version, v.commit_id)' 2>/dev/null)" = "0.0.0.dev0 None" ]; then
    # sglang/_version.py is written by the build backend (setuptools-scm). The authors built one role with a newer
    # setuptools-scm (no date cutoff on the build backend), whose template differs; the values are the same.
    echo "$role: $(( $(wc -l < $S/${role}_sgl.sha256) - 1 )) sglang files hash exactly as the authors' build; the generated sglang/_version.py differs only in its template (version 0.0.0.dev0, commit None, as the authors')"
  else echo "$role: SGLANG FILES DIFFER from the authors' build"; ok=0; fi
  r=$(cd $S && $V/bin/python3 -I -c "import sglang, torch; print('$role import ok', sglang.__version__, torch.__version__, torch.version.cuda)" 2>&1 | tail -1); echo "$r"
  case "$r" in *"import ok"*) ;; *) echo "$role: import check failed"; ok=0;; esac
done
diff $S/parent_sgl.sha256 $S/fix_sgl.sha256 | grep '^[<>]' | awk '{print $3}' | sort -u > $S/files_differing.txt
echo "files differing parent vs fix: $(tr '\n' ' ' < $S/files_differing.txt)"
echo "venv sizes: $(du -sh $ISO_ROOT/venvs/${PR}_parent $ISO_ROOT/venvs/${PR}_fix | awk '{print $1}' | tr '\n' ' ')  time $(( $(date +%s) - t0 )) s"
[ $ok = 1 ] && { echo "SETUP_LOCK_OK $PR"; exit 0; } || fail "checks above"
