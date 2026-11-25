#!/bin/bash
# setup_lock.sh : build this package's parent and fix venvs exactly as the authors had them (CPU and network only).
# Package addition (not an authors' script); it replaces the authors' resolver run (setup_b3.sh / setup_sgl.sh, also
# shipped) by an install from the pinned, hashed lock env/<pr>_deps.lock, which was compiled from the authors' freeze.
#   1. uv-managed CPython 3.10.20 under $ISO_ROOT/.uv-python
#   2. per role (parent, fix): uv venv; all dependencies from the lock (--no-deps --require-hashes);
#      vLLM: the per-commit wheel from wheels.vllm.ai, sha256 checked;  SGLang: python/ of the commit's source tarball,
#      every file checked against the authors' env/<pr>_<role>_src.sha256, built by its setuptools backend, --no-deps
#   3. checks: uv pip freeze equals the authors' env/<pr>_<role>_freeze.txt (apart from the local path of the engine
#      line); the installed engine files hash exactly as the authors' env/<pr>_<role>_{vllm,sgl}.sha256; import check.
# Output: $ISO_ROOT/b3/venvs/<pr>_{parent,fix}, records in $ISO_ROOT/b3/setup/. Exit 0 only if every check passes.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../../env.sh"; . "$ISO_ROOT/env/build.env"
B3=$ISO_ROOT/b3; E=$ISO_ROOT/env; S=$B3/setup; mkdir -p $S/wheels $B3/venvs $B3/src
export UV_PYTHON_INSTALL_DIR=$ISO_ROOT/.uv-python UV_PYTHON_BIN_DIR=$ISO_ROOT/.uv-bin UV_CACHE_DIR=${UV_CACHE_DIR:-$ISO_ROOT/.uv-cache} UV_PYTHON_PREFERENCE=only-managed
fail() { echo "SETUP FAILED: $*"; exit 1; }
t0=$(date +%s); echo "== setup_lock $PR ($ENGINE) fix=$FIX parent=$PARENT cutoff=$XNEWER  $("$UV" --version)"
"$UV" python install 3.10.20 || fail "uv python install 3.10.20"
PY=$("$UV" python find 3.10.20) || fail "python 3.10.20 not found"; echo "interpreter $PY"
XNP=(); [ -n "${OVERRIDE_PKG:-}" ] && XNP=(--exclude-newer-package "$OVERRIDE_PKG=$OVERRIDE_DATE")
PKGDIR=vllm; [ "$ENGINE" = sglang ] && PKGDIR=sglang
SUF=vllm; [ "$ENGINE" = sglang ] && SUF=sgl
ok=1
for role in parent fix; do
  c=$PARENT; [ $role = fix ] && c=$FIX; V=$B3/venvs/${PR}_$role
  rm -rf "$V"; "$UV" venv -q --python "$PY" "$V" || fail "venv $role"
  VIRTUAL_ENV=$V "$UV" pip install -q --no-deps --require-hashes --exclude-newer $XNEWER "${XNP[@]}" -r $E/${PR}_deps.lock || fail "dependencies $role"
  if [ "$ENGINE" = vllm ]; then
    u=URL_$role; h=SHA_$role; f=FILE_$role; W=$S/wheels/${PR}_$role/${!f}; mkdir -p "$(dirname "$W")"
    if [ ! -f "$W" ] || [ "$(sha256sum < "$W" | cut -c1-64)" != "${!h}" ]; then
      curl -fsSL -o "$W.part" "${!u}" || fail "download wheel $role"; mv "$W.part" "$W"; fi
    [ "$(sha256sum < "$W" | cut -c1-64)" = "${!h}" ] || fail "wheel sha256 $role"
    VIRTUAL_ENV=$V "$UV" pip install -q --no-deps "$W" || fail "install wheel $role"
    # pyairports stub, as the authors' setup_b3.sh (old releases were deleted from PyPI; outlines imports it)
    SP=$V/lib/python3.10/site-packages; mkdir -p $SP/pyairports
    printf '# P2 stub: PyPI no longer serves old pyairports releases\n' > $SP/pyairports/__init__.py; printf 'AIRPORT_LIST = []\n' > $SP/pyairports/airports.py
  else
    u=URL_$role; d=$B3/src/sgl_$c
    if [ ! -f $d/.ok ]; then rm -rf $d; mkdir -p $d
      curl -fsSL -o $d.tgz "${!u}" || fail "download source $role"
      tar -C $d --strip-components=1 --wildcards -xzf $d.tgz "sglang-*/python/*" || fail "untar $role"; rm -f $d.tgz
      (cd $d && sha256sum --quiet -c $E/${PR}_${role}_src.sha256) || fail "source files differ from the authors' ($role)"
      # The authors' src record misses the 181 files whose names contain spaces (MoE tuning configs "...block_shape=[128, 128].json"),
      # so it is checked file by file only; the complete check is the installed-file comparison below (696 files).
      echo "$role: $(wc -l < $E/${PR}_${role}_src.sha256) recorded source files match; $(cd $d && find python -type f | wc -l) extracted"
      touch $d/.ok; fi
    VIRTUAL_ENV=$V "$UV" pip install -q --no-deps --exclude-newer $XNEWER "$d/python" || fail "install sglang $role"
  fi
  VIRTUAL_ENV=$V "$UV" pip freeze > $S/${PR}_${role}_freeze.txt
  norm() { sed -E 's#^(vllm|sglang) @ .*#\1 @ <local>#' "$1"; }
  if diff <(norm $E/${PR}_${role}_freeze.txt) <(norm $S/${PR}_${role}_freeze.txt) > $S/${PR}_${role}_freeze_vs_authors.diff; then echo "$role: freeze equals the authors' ($(wc -l < $S/${PR}_${role}_freeze.txt) packages)"
  else echo "$role: FREEZE DIFFERS from the authors' (see $S/${PR}_${role}_freeze_vs_authors.diff)"; ok=0; fi
  (cd $V/lib/python3.10/site-packages && find $PKGDIR -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $S/${PR}_${role}_$SUF.sha256
  if cmp -s $S/${PR}_${role}_$SUF.sha256 $E/${PR}_${role}_$SUF.sha256; then echo "$role: $(wc -l < $S/${PR}_${role}_$SUF.sha256) $PKGDIR files hash exactly as the authors' build"
  else echo "$role: $PKGDIR FILES DIFFER from the authors' build"; ok=0; fi
  chk="import $PKGDIR, torch; print('$role import ok', $PKGDIR.__version__, torch.__version__, torch.version.cuda)"
  r=$($V/bin/python3 -I -c "$chk" 2>&1 | tail -1); echo "$r"; case "$r" in *"import ok"*) ;; *) echo "$role: import check failed"; ok=0;; esac
done
diff $S/${PR}_parent_$SUF.sha256 $S/${PR}_fix_$SUF.sha256 | grep '^[<>]' | awk '{print $3}' | sort -u > $S/${PR}_files_differing.txt
echo "files differing parent vs fix: $(tr '\n' ' ' < $S/${PR}_files_differing.txt)"
echo "venv sizes: $(du -sh $B3/venvs/${PR}_parent $B3/venvs/${PR}_fix | awk '{print $1}' | tr '\n' ' ')  time $(( $(date +%s) - t0 )) s"
[ $ok = 1 ] && { echo "SETUP_LOCK_OK $PR"; exit 0; } || fail "checks above"
