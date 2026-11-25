#!/bin/bash
# mk_cuda129.sh <cuda dir> <download dir> : variant F of the ST task on an RTX 5090.
# Replaces <cuda dir> (the CUDA 12.8 redist tree made by fetch_base.sh) with NVIDIA's CUDA 12.9.1 redist
# (cuda_nvcc 12.9.86, cuda_cudart 12.9.79, cuda_cccl 12.9.27) plus every file of the 12.8 tree that the 12.9
# archives lack (for example the curand headers FlashInfer's sampling kernels include). Keeps the original as <cuda dir>128.
# <download dir> must hold the three *-archive.tar.xz files from
# https://developer.download.nvidia.com/compute/cuda/redist/ (check their sha256 against NVIDIA's redistrib_12.9.1.json).
set -e; C=$(realpath "${1:?cuda dir}"); DL=$(realpath "${2:?download dir}"); D=${C}129; rm -rf "$D"; mkdir -p "$D"
for f in cuda_nvcc-linux-x86_64-12.9.86 cuda_cudart-linux-x86_64-12.9.79 cuda_cccl-linux-x86_64-12.9.27; do tar -xJf "$DL/$f-archive.tar.xz" --strip-components=1 -C "$D"; done
ln -sfn lib "$D/lib64"
cd "$C"; n=0; while IFS= read -r -d '' f; do [ -e "$D/$f" ] || { mkdir -p "$D/$(dirname "$f")"; cp -a "$f" "$D/$f"; n=$((n+1)); }; done < <(find . -type f -print0)
cd "$(dirname "$C")"; mv "$C" "${C}128"; ln -s "$(basename "$D")" "$C"; echo "cuda 12.9 tree: $n files from the 12.8 tree; $("$C"/bin/nvcc --version | tail -1)"
