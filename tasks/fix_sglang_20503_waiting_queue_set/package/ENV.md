# Environment spec for SGLang #20503 (scheduler: the waiting-queue update builds the set of admitted requests once)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/sgl-project/sglang/pull/20503, merged 2026-03-18 |
| Fix (merge commit) | `a6c7bb54eb3aeb60df44c6087ab7c929376e2022` |
| Parent (first parent) | `20a23e3173b5b3bdb16f19451f55d448b3747fb0` |
| Changed by the PR | `python/sglang/srt/managers/scheduler.py` only, +2/-3 in `Scheduler.get_new_batch_prefill`. After a prefill admission, the parent removes the admitted requests from the waiting queue with `[x for x in self.waiting_queue if x not in set(can_run_list)]`, which builds the set of the R admitted requests again for each of the W waiting ones: O(W x R) Python per admission, in the scheduler's event loop. The fix builds the set once. Which requests are admitted does not change, so outputs are bit-identical. |

## SGLang sources

Each role's package is built from the commit's source tarball (`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`),
of which only `python/` is extracted and built by the backend its `pyproject.toml` declares (setuptools; nothing else
from the tree is run). Every extracted file must hash as in the authors' record `env/<role>_src.sha256` (2,162 files),
and the file count must match. This tree declares no Rust extension; the script's step that would drop one changes
nothing here. The two `pyproject.toml` files are identical (`env/pyproject.diff` is empty).

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 170 pinned packages with sha256 hashes, all from PyPI, the same for parent and fix. Main pins: torch
  2.9.1 (the CUDA 12.8 build, cuDNN 9.10.2.21), torchvision 0.24.1, torchaudio 2.9.1, sglang-kernel 0.4.0, triton 3.5.1,
  flashinfer-python / flashinfer-cubin 0.6.6, torchao 0.9.0, transformers 4.57.1, xgrammar 0.1.27, huggingface-hub
  0.36.2, numpy 2.2.6, setuptools 82.0.1.
- How the lock was made: the authors resolved `python/[srt]` of the parent with uv `--exclude-newer 2026-03-20` (merge
  date + 2 days), no overrides (on the main pod for the batch-6 screen, 2026-11-25, with the batch-5 script
  `setup_sgl.sh` unchanged). The fix venv is a copy of the parent's with only
  the sglang package swapped (`--no-deps`). The freezes are `env/{parent,fix}_freeze.txt` and differ only in the sglang
  line (`env/freeze.diff`). The parent freeze minus the `sglang @ file://...` line was compiled with `uv pip compile
  --no-deps --generate-hashes --exclude-newer 2026-03-20`.
- Checks done by `setup/setup_lock.sh`: source hashes; `uv pip freeze` equals the authors' freeze (except the local path
  of the sglang line); all installed `sglang/` files (non-`.pyc`) hash exactly as `env/{parent,fix}_sgl.sha256` (2,115
  files); `import sglang, torch`. Files that differ between parent and fix: `env/files_differing.txt`.
- **cuDNN check.** This SGLang has `check_torch_2_9_1_cudnn_compatibility`, which refuses to start on torch 2.9.1 with
  cuDNN below 9.15 (a Conv3d slowdown, pytorch#168167). Qwen2.5-0.5B has no convolution, so the harness sets
  `SGLANG_DISABLE_CUDNN_CHECK=1` in both arms, as the screen did.

## CUDA pieces (FlashInfer JIT)

`setup/fetch_cuda.sh` builds a CUDA tree from NVIDIA's CUDA 12.8 redist archives (nvcc 12.8.93, cudart 12.8.90, cccl
12.8.90; sha256 pinned) plus the `include/` trees of the CUDA 12.8 math-library wheels (nvidia-cublas-cu12 12.8.4.1,
curand 10.3.9.90, cusparse 12.5.8.93, cusolver 11.7.3.90, cufft 11.3.3.83; `env/cuda_headers.lock`, hash-pinned; the
same versions torch 2.9.1 depends on). This is the tree the screen used (`/tmp/c5y/st/cuda` on main, the C48 package's tree, built by
the same recipe as batch 4's `/tmp/b4/cuda`); its 1,740 files must hash as `env/cuda_tree.sha256`. The harness passes
`CUDA_HOME=$STAGE/cuda` to every run, so that FlashInfer can JIT-compile a module if this SGLang asks for one. Compile
caches (FlashInfer workspace, Triton, Inductor) live under `$STAGE/cache/<pr>/<role>/<host>-g<gpu>` and are kept between
runs of the same role.

## Model

`Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` (the snapshot the batch-6 screen
downloaded on 2026-11-25), whole repository (~1 GB), bf16. Its one LFS blob (sha256) is listed in `env/build.env`
(`MODEL_LFS`). `get_model.sh` writes `refs/main` = that revision, because the engine resolves the model offline through
it.

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4), main pod of 4090-server (Threadripper PRO 5975WX, 4 CCDs),
  gcc 12.2. One CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU2 16-23,48-55, GPU3 24-31,56-63 (GPU0
  0-7,32-39, GPU1 8-15,40-47). The screen ran on the same pod, GPU2.
- Venvs, model, CUDA tree and compile caches staged on local NVMe (`$STAGE`); the persistent copies on NFS.

## RTX 5090 (sm_120): not run

torch 2.9.1+cu128 carries sm_120. Whether sglang-kernel 0.4.0 carries sm_120 code for the kernels this load uses was not
checked. The gain is scheduler Python per admission, so its size depends on the CPU more than on the GPU, and would need
its own calibration, after the 5090 host's driver-hang issue is ruled out.
