# Environment spec for SGLang #27868 (Qwen3.5 decode: GDN in-projections and Q/K norm on a second CUDA stream)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/sgl-project/sglang/pull/27868, merged 2026-06-15 |
| Fix (merge commit) | `d5899b95c4d54db58d5412b9ce0cc487b212d9a0` |
| Parent (first parent) | `09e9c4fde3b4326c4a66fbb014d9c7a6b8995b30` |
| Changed by the PR | `python/sglang/srt/models/qwen3_5.py` only. Two module flags, `_gdn_use_alt_stream` and `_qknorm_use_alt_stream`, become true on every CUDA build (before: only on ROCm with an environment variable), and the decoder layers pass the model's second CUDA stream to the attention and GDN blocks on CUDA. Effect: inside CUDA-graph capture (decode), the GDN layers run the small `in_proj_ba` GEMM on the side stream next to `in_proj_qkvz`, and the full-attention layers run the K RMSNorm next to the Q RMSNorm. The graph then holds two parallel branches; the arithmetic is unchanged, so outputs are bit-identical. |

## SGLang sources

Each role's package is built from the commit's source tarball (`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`),
of which only `python/` is extracted and built by the backend its `pyproject.toml` declares (setuptools; nothing else
from the tree is run). Every extracted file must hash as in the authors' record `env/<role>_src.sha256` (3,106 files),
and the file count must match. Then, as the authors did in batch 4, the `[[tool.setuptools-rust.ext-modules]]` block
(the gRPC server's Rust core) is removed from `pyproject.toml` in both roles, with `SGLANG_BUILD_RUST_EXTS=none`: the
pods have no cargo, and the offline `Engine` never imports the gRPC core. The two `pyproject.toml` files are identical
(`env/pyproject.diff` is empty).

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 196 pinned packages with sha256 hashes, the same for parent and fix. Main pins: torch 2.11.0+cu129,
  torchvision 0.26.0+cu129, torchaudio 2.11.0+cu129 (direct wheel URLs on download-r2.pytorch.org), sglang-kernel
  0.4.3+cu129 (GitHub release wheel), torchao 0.17.0+cu129, triton 3.6.0, flashinfer-python / flashinfer-cubin 0.6.12,
  nvidia-cutlass-dsl 4.5.2, transformers 5.8.1, xgrammar 0.2.1, huggingface-hub 1.19.0, numpy 2.2.6, setuptools 81.0.0.
- How the lock was made: the authors (batch 4, candidate C56) resolved `python/` of the parent with uv
  `--exclude-newer 2026-06-17` (merge date + 2 days) and the overrides in `env/override.txt`: this SGLang asks for torch
  2.11 whose default PyPI wheels are built for CUDA 13, which the 4090 hosts' driver (CUDA 12.4) cannot run, so torch,
  torchvision, torchaudio and sglang-kernel were pinned to their cu129 builds and `cuda-bindings` to 12.x. The fix venv
  is a copy of the parent's resolution with only the sglang package swapped (`--no-deps`). Their freezes are
  `env/{parent,fix}_freeze.txt` (they differ only in the sglang line, `env/freeze.diff`). On 2026-11-24 the parent freeze
  minus the `sglang @ file://...` line was compiled with `uv pip compile --no-deps --generate-hashes --exclude-newer
  2026-06-17` plus per-package exceptions for the five override packages (their pinned builds were uploaded later than
  the cutoff; the exceptions only allow exactly the versions in the freeze) and the PyTorch cu129 index. The PyTorch index
  served `jinja2` and `markupsafe` without hashes, so their hashes were taken from PyPI (same versions).
- After installing the lock, `nvidia-cutlass-dsl-libs-cu13` is uninstalled in both venvs (as the authors did on
  2026-11-24): with it, the CuTe-DSL loads CUDA 13 libraries and FlashInfer's Gemma RMSNorm fails on a CUDA 12.4 driver;
  without it the DSL uses `nvidia-cutlass-dsl-libs-base`. The authors' freezes were taken before that removal, so the
  freeze comparison ignores that one line.
- Checks done by `setup/setup_lock.sh`: source hashes; `uv pip freeze` equals the authors' freeze (except the local path
  of the sglang line and the removed package); all installed `sglang/` files (non-`.pyc`) hash exactly as
  `env/{parent,fix}_sgl.sha256` (3,064 files); `import sglang, torch`. Files that differ between parent and fix:
  `env/files_differing.txt`.

## CUDA pieces (FlashInfer JIT)

This SGLang compiles FlashInfer's norm module the first time the engine captures CUDA graphs, with `nvcc` from
`$CUDA_HOME` and a host `g++`. `setup/fetch_cuda.sh` builds that tree from NVIDIA's CUDA 12.8 redist archives (nvcc
12.8.93, cudart 12.8.90, cccl 12.8.90; sha256 pinned) plus the `include/` trees of the CUDA 12.8 math-library wheels
(nvidia-cublas-cu12 12.8.4.1, curand 10.3.9.90, cusparse 12.5.8.93, cusolver 11.7.3.90, cufft 11.3.3.83;
`env/cuda_headers.lock`, hash-pinned). That is the tree the authors used in batch 4 (`/tmp/b4/cuda`); its 1,740 files must
hash as `env/cuda_tree.sha256`. The harness passes `CUDA_HOME=$STAGE/cuda` to every run. Compile caches (FlashInfer
workspace, Triton, Inductor) live under `$STAGE/cache/<pr>/<role>/<host>-g<gpu>` and are kept between runs of the same
role (the scored region is the steady decode after start-up, so cold vs warm caches only change start-up time).

## Model

`Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (the snapshot batch 4 used), whole repository
(~9 GB), bf16. LFS blobs (sha256) are listed in `env/build.env` (`MODEL_LFS`). `get_model.sh` writes `refs/main` = that
revision, because the engine resolves the model offline through it. Qwen3.5-4B is a hybrid model: most layers are
Gated DeltaNet (linear attention), every fourth a full-attention layer.

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4; the cu129 torch wheels run on it through CUDA minor-version
  compatibility), main pod of 4090-server (Threadripper PRO 5975WX, 4 CCDs), gcc 12.2. One CCD (8 cores + SMT siblings)
  per GPU job via `taskset`: GPU0 0-7,32-39, GPU1 8-15,40-47, GPU2 16-23,48-55, GPU3 24-31,56-63.
- Venvs, model, CUDA tree and compile caches staged on local NVMe (`$STAGE`); the persistent copies on NFS.

## RTX 5090 (sm_120): what a run there would need (not run)

Checked on CPU by reading the wheels' fatbin sections (2026-11-24, `c5y_work/pod/scripts/fatbin_arch.py`):
- torch 2.11.0+cu129 carries sm_120 (`_cuda_getArchFlags()`: sm_75, 80, 86, 90, 100, 120, compute_120). Triton 3.6.0
  generates sm_120 code.
- sglang-kernel 0.4.3+cu129: `common_ops.abi3.so` (both the sm90 and sm100 builds) carries SASS for sm_80/89/90/100/120,
  so RMSNorm, SiLU-and-mul and rotary run. `flash_ops.abi3.so` (FlashAttention 3) has sm_80/86/90 only, but this load
  uses the triton attention backend. FlashInfer's norm module is JIT-compiled by `nvcc` 12.8, which supports sm_120.
- So the pinned environment should run unchanged on a 5090. Needed: a driver for CUDA >= 12.9 (the cu129 wheels; R575 or
  newer), its own calibration (REF and A/A are hardware-specific; the side-stream overlap may gain more or less on a
  GPU with more SMs), and the 5090 host's known driver-hang issue fixed or ruled out first (stock wheels, not generated
  kernels, but the same driver).
