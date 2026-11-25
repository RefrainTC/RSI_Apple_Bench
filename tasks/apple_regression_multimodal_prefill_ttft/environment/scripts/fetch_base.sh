#!/usr/bin/env bash
# fetch_base.sh : pinned downloads shared by the MM and ST samples. CPU + network only, no root, idempotent.
# (Same file in both scripts/ dirs.) Everything lands under $RSI_ROOT (required):
#   bin/uv                                 uv 0.11.14, GitHub release tarball (sha256 checked)
#   uvpy/cpython-3.12.13-linux-x86_64-gnu  CPython 3.12.13, uv's standalone build "May 10 2026" (`uv python install --no-bin`)
#   mm/cuda                                CUDA 12.8 redist: nvcc 12.8.93, cudart 12.8.90, cccl 12.8.90 (sha256 checked)
#   apple                                  Apple ml-rsi-inference-bench @1a38072f (untrusted download: task dirs are copied
#                                          out of it; its Python only runs as the task's own harness or with -I)
set -euo pipefail
D=${RSI_ROOT:?set RSI_ROOT}
mkdir -p "$D/bin" "$D/uvpy" "$D/mm" "$D/dl"
export TMPDIR=$D/dl
say() { echo "== $(date '+%F %T') $*"; }
fetch() {  # <url> <out file> <sha256>
  local i
  for i in 1 2 3; do
    curl -fsSL --retry 5 -o "$2" "$1" && echo "$3  $2" | sha256sum -c --quiet && return 0
    echo "retry $2"; rm -f "$2"; sleep 5
  done
  return 1
}

if [ ! -x "$D/bin/uv" ]; then
  say "uv 0.11.14"
  fetch https://github.com/astral-sh/uv/releases/download/0.11.14/uv-x86_64-unknown-linux-gnu.tar.gz "$D/dl/uv.tgz" \
    f3b623eb0e6141a7053d571d59a0bdc341e0f238ea8f5f0b4815ddbec9a2a296
  tar -xzf "$D/dl/uv.tgz" -C "$D/dl" && mv "$D/dl/uv-x86_64-unknown-linux-gnu/uv" "$D/bin/uv"
fi
"$D/bin/uv" --version

PY=$D/uvpy/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
if [ ! -x "$PY" ]; then
  say "CPython 3.12.13 (uv standalone build)"
  UV_PYTHON_INSTALL_DIR=$D/uvpy UV_CACHE_DIR=$D/dl/uv-cache "$D/bin/uv" python install --no-bin 3.12.13
fi
"$PY" -I -c "import sys; print(sys.version); assert sys.version.startswith('3.12.13 (main, May 10 2026'), 'unexpected build'"

if [ ! -x "$D/mm/cuda/bin/nvcc" ]; then
  say "CUDA 12.8 redist"
  RD=https://developer.download.nvidia.com/compute/cuda/redist
  rm -rf "$D/mm/cuda.partial"; mkdir -p "$D/mm/cuda.partial"
  while read -r sha p; do
    f=$D/dl/$(basename "$p")
    [ -f "$f" ] && echo "$sha  $f" | sha256sum -c --quiet 2>/dev/null || fetch "$RD/$p" "$f" "$sha"
    tar -xJf "$f" --strip-components=1 -C "$D/mm/cuda.partial"
  done <<'EOF'
9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3 cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68 cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz
0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz
EOF
  ln -sfn lib "$D/mm/cuda.partial/lib64"
  mv "$D/mm/cuda.partial" "$D/mm/cuda"
fi
"$D/mm/cuda/bin/nvcc" --version | tail -1

APPLE_SHA=1a38072fbe284234c874356b86c9df666a5a79a3
if [ "$(git -C "$D/apple" rev-parse HEAD 2>/dev/null)" != "$APPLE_SHA" ]; then
  say "Apple ml-rsi-inference-bench @$APPLE_SHA"
  rm -rf "$D/apple"; mkdir -p "$D/apple"
  git -C "$D/apple" init -q
  git -C "$D/apple" remote add origin https://github.com/apple-aiml-research/ml-rsi-inference-bench.git
  git -C "$D/apple" fetch -q --depth 1 origin "$APPLE_SHA"
  git -C "$D/apple" checkout -q FETCH_HEAD
fi
[ "$(git -C "$D/apple" rev-parse HEAD)" = "$APPLE_SHA" ]
say "FETCH_BASE DONE (uv, CPython 3.12.13, CUDA 12.8, Apple repo @${APPLE_SHA:0:8})"
