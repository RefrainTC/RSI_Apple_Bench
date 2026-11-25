# Environment spec for SGLang #20972 (no blocking copy when logprobs are returned)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/sgl-project/sglang/pull/20972, merged 2026-03-27 |
| Fix (merge commit) | `a27651d5e0d138d2a5ddc529ee95317fd57328fa` |
| Parent (first parent) | `e2b8463c80367730db57d5c38c8835ab326ab468` |
| Changed by the PR | `python/sglang/srt/layers/sampler.py` (`get_top_logprobs` / `get_token_ids_logprobs` called with `no_copy_to_cpu=True` in the forward thread) and `python/sglang/srt/managers/scheduler_output_processor_mixin.py` (the `.tolist()` moves to the result processing, after the copy-done event, for prefill and decode) |

## SGLang sources

SGLang has no per-commit wheels. Each role's package is built from the commit's source tarball
(`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`), of which only `python/` is extracted and built by the
backend its `pyproject.toml` declares (with `SGLANG_BUILD_RUST_EXTS=none`; nothing else from the tree is run). GitHub
does not guarantee byte-stable tarballs, so the check is per file: every extracted file must hash as in the authors'
record `env/<role>_src.sha256`, and the file count must match.

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 176 pinned packages with sha256 hashes, the same for parent and fix. Main pins: torch 2.9.1 (cu128 wheels from PyPI), torchvision 0.24.1, triton 3.5.1, sglang-kernel 0.4.0 (module `sgl_kernel`), flashinfer-python / flashinfer-cubin 0.6.6, transformers 5.3.0, xgrammar 0.1.32, huggingface-hub 1.8.0, cuda-python 12.9.0, setuptools 82.0.1.
- How the lock was made: the authors (batch 4, `SCREEN_BATCH4.md`, candidate C53) resolved `python/[srt]` of the parent
  with uv `--exclude-newer 2026-03-29` (merge date + 2 days) from PyPI only; the fix venv was a copy of the parent venv
  with only the sglang package swapped (`--no-deps`), after checking that the two `pyproject.toml` files are identical.
  Their freezes are `env/{parent,fix}_freeze.txt` (they differ only in the sglang line, `env/freeze.diff`). On
  2026-11-24 the parent freeze minus the `sglang @ file://...` line (and nothing else: packages whose names merely start
  with `sglang` or `sgl` stay) was compiled with `uv pip compile --no-deps --generate-hashes --exclude-newer 2026-03-29
  --python-version 3.10 --python-platform x86_64-unknown-linux-gnu`.
- Checks done by `setup/setup_lock.sh`: `uv pip freeze` equals the authors' freeze (except the local path of the sglang
  line); all installed `sglang/` files (non-`.pyc`) hash exactly as `env/{parent,fix}_sgl.sha256`; `import sglang, torch`.
  Files that differ between parent and fix: `sglang/srt/layers/sampler.py` and `sglang/srt/managers/scheduler_output_processor_mixin.py` (`env/files_differing.txt`).

## CUDA pieces (FlashInfer JIT)

SGLang builds of 2026 compile FlashInfer's RMSNorm module with nvcc during CUDA-graph capture. Machines without a CUDA
toolkit (the authors' pods) need `setup/fetch_cuda.sh`: NVIDIA's CUDA 12.8 redist archives (nvcc 12.8.93, cudart 12.8.90,
cccl 12.8.90; sha256-pinned in the script) plus the include directories of the venv's own `nvidia-cublas-cu12`,
`nvidia-curand-cu12`, `nvidia-cusparse-cu12`, `nvidia-cusolver-cu12` and `nvidia-cufft-cu12` wheels (the module includes
`cublasLt.h` and `curand_kernel.h`). The tree must hash as the authors' (`env/cuda_tree.sha256`, 1740 files). Jobs get
`CUDA_HOME=$STAGE/cuda`; the compiled module goes to a per-role, per-GPU cache under `$STAGE/cache` (the first run on a
GPU compiles it during engine start-up, ~15-20 s extra, untimed). A host `g++` is needed.

## Model

`Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (the same snapshot as the other P2
SGLang tasks), whole repository (3.1 GB), bf16. `model.safetensors` sha256
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`. `get_model.sh` writes `refs/main` = that revision,
because the engine resolves the model offline through it.

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4; the cu128 torch wheels run on it), n1 pod of 4090-server
  (Threadripper PRO 5975WX). One CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU0 0-7,32-39, GPU1 24-31,56-63.
- Venvs, model, CUDA tree and compile caches staged on local NVMe (`$STAGE`); the persistent copies were on NFS.

## RTX 5090 (sm_120): what a run there would need (not run)

Checked on CPU by reading the wheels' fatbin sections (2026-11-24): torch 2.9.1+cu128 carries sm_120 SASS
(`torch._C._cuda_getArchFlags()`: sm_70 ... sm_90, sm_100, sm_120); the `sgl_kernel` `common_ops` and `spatial_ops`
libraries carry sm_80/89/90/100/120 SASS; `flash_ops` (FlashAttention 3, sm_80/86/90 only) and `flashmla_ops` (sm_90/100)
have no sm_120 code but are not used by this task's triton attention backend. FlashInfer's JIT module is compiled by
nvcc 12.8, which targets sm_120; Triton 3.5.1 supports sm_120. So the software stack has sm_120 code for every kernel
this load runs. A 5090 run needs a driver for CUDA >= 12.8 (Blackwell needs R570 or newer), its own calibration (REF,
A/A and the per-card offset are hardware-specific), and the 5090 host's known driver-hang issue to be fixed or ruled out
first (P2 notes: generated kernels deadlocked nvidia_uvm there; these are stock wheels, not generated kernels, but the
same driver).

