#!/bin/bash
# setup_sgl.sh <pr> <fixcommit> <parentcommit> <mergedate> : SGLang per-commit venvs (batch 3). Source tarballs from
# codeload.github.com (untrusted data: only the python/ package is built, by the setuptools backend declared in its
# pyproject.toml; nothing from the tree is run otherwise). Parent venv = python/[srt] at the parent commit, deps as of merge
# date + 2 days; fix venv = the same, pinned to the parent's freeze. Markers b3/setup/done_<pr> / failed_<pr>.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../../env.sh"; R=$ISO_ROOT; P2=$ISO_ROOT; B3=$P2/b3; S=$B3/setup
export UV_PYTHON_INSTALL_DIR=$R/.uv-python UV_PYTHON_BIN_DIR=$R/.uv-bin UV_CACHE_DIR=$P2/.uv-cache UV_PYTHON_PREFERENCE=only-managed
PY=$($UV python find 3.10.20)
pr=$1; F=$2; P=$3; D=$4; X=$(date -d "$D + 2 days" +%F); mkdir -p $S $B3/src $B3/venvs; cd $S; rm -f done_$pr failed_$pr
fail() { echo "FAILED $pr: $1"; echo "$1" > $S/failed_$pr; exit 1; }
echo "=== SGLang PR $pr fix=$F parent=$P exclude-newer=$X $(date -u +%T)"
for role in parent fix; do c=$P; [ $role = fix ] && c=$F; d=$B3/src/sgl_$c
  if [ ! -f $d/.ok ]; then rm -rf $d; mkdir -p $d
    curl -fsSL -H "User-Agent: b3-screen" -o $d.tgz https://codeload.github.com/sgl-project/sglang/tar.gz/$c || fail "download $role"
    tar -C $d --strip-components=1 --wildcards -xzf $d.tgz "sglang-*/python/*" || fail "untar $role"; rm -f $d.tgz; (cd $d && find python -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $S/${pr}_${role}_src.sha256; touch $d/.ok
  fi
  rm -rf $B3/venvs/${pr}_$role; $UV venv -q --python $PY $B3/venvs/${pr}_$role
done
# flashinfer-python 0.2.6.post1 is an sdist that needs nvcc to build; the pods have no CUDA toolkit. It is left out
# (override marker "never"); the screen uses the triton attention backend and pytorch sampling, which do not import it.
printf 'flashinfer-python ; sys_platform == "never"\n' > $S/sgl_overrides.txt
VIRTUAL_ENV=$B3/venvs/${pr}_parent $UV pip install -q --exclude-newer $X --override $S/sgl_overrides.txt "$B3/src/sgl_$P/python[srt]" setuptools || fail "install parent"
VIRTUAL_ENV=$B3/venvs/${pr}_parent $UV pip freeze | grep -v -i "^sglang" > ${pr}_constraints.txt
VIRTUAL_ENV=$B3/venvs/${pr}_fix $UV pip install -q --exclude-newer $X --override $S/sgl_overrides.txt -c ${pr}_constraints.txt "$B3/src/sgl_$F/python[srt]" setuptools || fail "install fix"
for role in parent fix; do VIRTUAL_ENV=$B3/venvs/${pr}_$role $UV pip freeze > ${pr}_${role}_freeze.txt; done
diff ${pr}_parent_freeze.txt ${pr}_fix_freeze.txt > ${pr}_freeze.diff; echo "freeze diff lines: $(grep -c '^[<>]' ${pr}_freeze.diff)"
grep -i -E "^(torch|sgl-kernel|flashinfer-python|triton|transformers)==" ${pr}_parent_freeze.txt | tr '\n' ' '; echo
for role in parent fix; do (cd $B3/venvs/${pr}_$role/lib/python3.10/site-packages && find sglang -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > ${pr}_${role}_sgl.sha256; done
diff ${pr}_parent_sgl.sha256 ${pr}_fix_sgl.sha256 | grep '^[<>]' | awk '{print $3}' | sort -u > ${pr}_sgl_pkg_files.txt
echo "sglang files differing: $(tr '\n' ' ' < ${pr}_sgl_pkg_files.txt)"
ok=1; for role in parent fix; do $B3/venvs/${pr}_$role/bin/python3 -I -c "import sglang, torch, sgl_kernel; print('$role import ok', sglang.__version__, torch.__version__, torch.version.cuda)" 2>&1 | tail -1 | tee /dev/stderr | grep -q "import ok" || ok=0; done
[ $ok = 1 ] && { date -u +%FT%T > $S/done_$pr; echo "PR_DONE $pr"; } || fail "import check"
