# Reference solution: SGLang #27868

- Upstream PR: https://github.com/sgl-project/sglang/pull/27868, merged 2026-06-15.
- Fix (merge commit): `d5899b95c4d54db58d5412b9ce0cc487b212d9a0`.
- Parent (first parent, the task's baseline): `09e9c4fde3b4326c4a66fbb014d9c7a6b8995b30`.

## What it changes

One file: `python/sglang/srt/models/qwen3_5.py`.

Qwen3.5 already had code to overlap two small pieces of each decoder layer on a second CUDA stream, but it was
switched on only for ROCm, with an environment variable. An earlier AMD-focused PR had gated it behind `_is_hip`. The
fix turns it on for every CUDA build. Inside CUDA-graph capture (decode steps):
- the Gated-DeltaNet layers run their small `in_proj_ba` GEMM on the side stream next to the large `in_proj_qkvz`
  GEMM;
- the full-attention layers run the K RMSNorm next to the Q RMSNorm.

The captured graph then holds two parallel branches per layer. No arithmetic changes, so outputs are bit-identical to
the parent's. At batch 4 this saves ~0.3 ms per decode step; at batch 1 almost nothing (1.006x).

## How the package builds the fixed arm

- Each arm's SGLang is built from the commit's source tarball (`python/` only). The Rust gRPC extension block is
  dropped from `pyproject.toml`, identically in both arms. Every extracted file must hash as in the authors' record.
- Dependencies come from one hashed lock (`package/env/deps.lock`), as of the merge date + 2 days, with the cu129
  builds of torch 2.11.0 and sglang-kernel 0.4.3 (the default torch 2.11 wheels need CUDA 13, and the hosts' driver is
  CUDA 12.4); triton 3.6.0, flashinfer 0.6.12, transformers 5.8.1. `nvidia-cutlass-dsl-libs-cu13` is removed after
  install.
- The fix venv is the parent's freeze with only the sglang package swapped. Only `sglang/srt/models/qwen3_5.py`
  differs (`package/env/files_differing.txt`).
- `package/setup/fetch_cuda.sh` downloads a CUDA 12.8 `nvcc` + headers tree for FlashInfer's JIT. On an RTX 5090 this
  must be a CUDA 12.9 tree instead (see `../README.md`).
- `package/setup/setup_lock.sh` builds both venvs and checks them against the authors' build.
  `package/harness/run.sh <label> score` (candidate `fix` by default) scores this arm.

## Result

REF **1.0276** on RTX 4090 (9 jobs, 1.0261-1.0291); 1.058 on RTX 5090 with a CUDA 12.9 tree (3 jobs). The fix passes
every check, bit-exact. See `../README.md` and `../results/summary.json`.
