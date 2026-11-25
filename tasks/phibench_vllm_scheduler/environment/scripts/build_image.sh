#!/usr/bin/env bash
# build_image.sh : build the RTX 5090 variant of the task's single-shot (:oss) image. CPU and network only.
# Needs $PHI_TASK = a copy of Phi-Bench's tasks/e2e/vllm-scheduler-mixed-batch-serving/task with
# environment/Dockerfile.5090 placed in it (see README). Dockerfile.5090 differs from the task's Dockerfile in 4 lines
# (diffs/dockerfile_5090.diff): CUDA 12.8.1 base, TORCH_CUDA_ARCH_LIST 12.0, and cu126 -> cu128 for torch and vLLM.
# We used the classic builder (DOCKER_BUILDKIT=0, no buildx on our host); the Dockerfile uses no BuildKit-only syntax.
# Add --no-cache when rebuilding on a host that already has layers from an earlier build.
set -euo pipefail
T=${PHI_TASK:?set PHI_TASK to the task copy}
IMG=${IMG:-phibench-vllm-sched:oss-sm120}
ARCH=${ARCH:-12.0}
t0=$(date +%s)
cd "$T"
DOCKER_BUILDKIT=0 docker build ${NO_CACHE:+--no-cache} --build-arg TORCH_CUDA_ARCH_LIST=$ARCH -f environment/Dockerfile.5090 -t "$IMG" .
echo "[build] done in $(( $(date +%s) - t0 ))s"
docker image inspect "$IMG" --format '{{.Id}} {{.Size}}'
