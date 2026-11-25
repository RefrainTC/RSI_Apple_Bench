#!/bin/bash
# fetch_cuda.sh : the CUDA pieces FlashInfer's JIT needs at engine start (this SGLang compiles FlashInfer's norm module
# during CUDA-graph capture; machines without a CUDA toolkit need this). Into $ISO_ROOT/cuda:
#   NVIDIA's CUDA 12.8 redist archives (nvcc 12.8.93, cudart 12.8.90, cccl 12.8.90; sha256 pinned), plus the include/
#   trees of the CUDA-12.8 nvidia-cublas / curand / cusparse / cusolver / cufft wheels (hash-pinned in
#   env/cuda_headers.lock; the norm module includes cublasLt.h and curand_kernel.h). This is the tree the authors used in
#   batch 4 (/tmp/b4/cuda); the result must hash as env/cuda_tree.sha256. Also needs a host C++ compiler (gcc 12.2 on the
#   authors' pods).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
D=$ISO_ROOT; RD=https://developer.download.nvidia.com/compute/cuda/redist
fail() { echo "FETCH_CUDA FAILED: $*"; exit 1; }
export UV_CACHE_DIR=${UV_CACHE_DIR:-$ISO_ROOT/.uv-cache}
PY=$ISO_ROOT/venvs/${PR}_parent/bin/python3; [ -x "$PY" ] || fail "build the venvs first"
rm -rf $D/cuda.partial $D/cuda.hdr; mkdir -p $D/cuda.partial $D/cuda.dl
while read sha p; do f=$D/cuda.dl/$(basename $p)
  if [ ! -f $f ] || ! echo "$sha  $f" | sha256sum -c --quiet > /dev/null 2>&1; then
    # long-lived HTTPS connections can stall: resume with a fresh connection whenever speed drops below 1 MB/s for 8 s
    for k in $(seq 1 200); do curl -fsSL -C - --speed-limit 1000000 --speed-time 8 -o $f $RD/$p && break; done
  fi
  echo "$sha  $f" | sha256sum -c --quiet || fail "sha256 $p"
  tar -xJf $f --strip-components=1 -C $D/cuda.partial || fail "untar $p"
done <<'L'
9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3 cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68 cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz
0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz
L
"$UV" pip install -q --no-deps --require-hashes --python "$PY" --target $D/cuda.hdr -r $ISO_ROOT/env/cuda_headers.lock || fail "header wheels"
for lib in cublas curand cusparse cusolver cufft; do cp -r $D/cuda.hdr/nvidia/$lib/include/. $D/cuda.partial/include/ || fail "headers $lib"; done
rm -rf $D/cuda.hdr; ln -sfn lib $D/cuda.partial/lib64
(cd $D/cuda.partial && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > $D/cuda_tree.sha256
if cmp -s $D/cuda_tree.sha256 $ISO_ROOT/env/cuda_tree.sha256; then echo "cuda tree: $(wc -l < $D/cuda_tree.sha256) files hash exactly as the authors'"
else diff $ISO_ROOT/env/cuda_tree.sha256 $D/cuda_tree.sha256 | head; fail "cuda tree differs from the authors'"; fi
rm -rf $D/cuda; mv $D/cuda.partial $D/cuda; rm -rf $D/cuda.dl
$D/cuda/bin/nvcc --version | tail -1; echo FETCH_CUDA_OK
