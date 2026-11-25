# Environment spec for SGLang #14956 (Qwen2-VL decode: mrope positions of text requests from the CPU copy of the sequence lengths)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/sgl-project/sglang/pull/14956, merged 2025-12-13 |
| Fix (merge commit) | `69cfb17b9a5bfde2b6b150989159a2942ef21b89` |
| Parent (first parent) | `a7a4b1755d1b6eb7b04925f8453647368c029487` |
| Changed by the PR | `python/sglang/srt/model_executor/forward_batch_info.py` only, function `_compute_mrope_positions` (models with multimodal rotary positions, such as Qwen2-VL). In the decode branch, the parent builds the 3 x 1 position of a request without image with `torch.full((3, 1), self.seq_lens[i] - 1, device=cuda)`. `self.seq_lens` is a GPU tensor, so the fill value is a 0-d CUDA tensor that `torch.full` reads back: one blocking device-to-host copy per text request per decode step, in the forward thread. The fix reads `self.seq_lens_cpu` (and takes the batch size from it), so no copy waits for the GPU. The value is the same, so outputs are bit-identical. |

## SGLang sources

Each role's package is built from the commit's source tarball (`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`),
of which only `python/` is extracted and built by the backend its `pyproject.toml` declares (setuptools; nothing else
from the tree is run). Every extracted file must hash as in the authors' record `env/<role>_src.sha256` (1,470 files),
and the file count must match. This tree declares no Rust extension; the script's step that would drop one changes
nothing here. The two `pyproject.toml` files are identical (`env/pyproject.diff` is empty).

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 165 pinned packages with sha256 hashes, all from PyPI, the same for parent and fix. Main pins: torch
  2.9.1 (the CUDA 12.8 build, cuDNN 9.10.2.21), torchvision 0.24.1, torchaudio 2.9.1, sgl-kernel 0.3.19, triton 3.5.1,
  flashinfer-python / flashinfer-cubin 0.5.3, torchao 0.9.0, transformers 4.57.1, xgrammar 0.1.27, huggingface-hub
  0.36.0, numpy 2.2.6, setuptools 80.9.0.
- How the lock was made: the authors resolved `python/[srt]` of the parent with uv `--exclude-newer 2025-12-15` (merge
  date + 2 days), no overrides (first on the n1 pod for the batch-5 screen, then again on the main pod on 2026-11-25 with
  the same script; both freezes and all sglang file hashes are equal). The fix venv is a copy of the parent's with only
  the sglang package swapped (`--no-deps`). The freezes are `env/{parent,fix}_freeze.txt` and differ only in the sglang
  line (`env/freeze.diff`). The parent freeze minus the `sglang @ file://...` line was compiled with `uv pip compile
  --no-deps --generate-hashes --exclude-newer 2025-12-15`.
- Checks done by `setup/setup_lock.sh`: source hashes; `uv pip freeze` equals the authors' freeze (except the local path
  of the sglang line); all installed `sglang/` files (non-`.pyc`) hash exactly as `env/{parent,fix}_sgl.sha256` (1,432
  files); `import sglang, torch`. Files that differ between parent and fix: `env/files_differing.txt`.
- **cuDNN check.** This SGLang has `check_torch_2_9_1_cudnn_compatibility`, which refuses to start on torch 2.9.1 with
  cuDNN below 9.15 (a Conv3d slowdown, pytorch#168167; the vision tower uses Conv3d). The load is text-only, so the
  harness sets `SGLANG_DISABLE_CUDNN_CHECK=1` in both arms, as the screen did. The image transfer load (`tr_img`) does
  run the vision tower in both arms with that slow Conv3d; it measures this PR's decode path only (no gain expected).

## CUDA pieces (FlashInfer JIT)

`setup/fetch_cuda.sh` builds a CUDA tree from NVIDIA's CUDA 12.8 redist archives (nvcc 12.8.93, cudart 12.8.90, cccl
12.8.90; sha256 pinned) plus the `include/` trees of the CUDA 12.8 math-library wheels (nvidia-cublas-cu12 12.8.4.1,
curand 10.3.9.90, cusparse 12.5.8.93, cusolver 11.7.3.90, cufft 11.3.3.83; `env/cuda_headers.lock`, hash-pinned; the
same versions torch 2.9.1 depends on). This is the tree the screen used (`/tmp/b5/cuda` on n1, built by the same recipe
as batch 4's `/tmp/b4/cuda`); its 1,740 files must hash as `env/cuda_tree.sha256`. The harness passes
`CUDA_HOME=$STAGE/cuda` to every run, so that FlashInfer can JIT-compile a module if this SGLang asks for one. Compile
caches (FlashInfer workspace, Triton, Inductor) live under `$STAGE/cache/<pr>/<role>/<host>-g<gpu>` and are kept between
runs of the same role.

## Model

`Qwen/Qwen2-VL-2B-Instruct`, revision `1860c46f12b7449d0f5b426a7474ab715342bc65`, whole repository (~4.4 GB), bf16. LFS
blobs (sha256) are listed in `env/build.env` (`MODEL_LFS`). `get_model.sh` writes `refs/main` = that revision, because the
engine resolves the model offline through it. Its language model uses multimodal rotary positions (mrope), which is
why the decode step goes through `_compute_mrope_positions` even for text-only requests.

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4), main pod of 4090-server (Threadripper PRO 5975WX, 4 CCDs),
  gcc 12.2. One CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU2 16-23,48-55, GPU3 24-31,56-63 (GPU0
  0-7,32-39, GPU1 8-15,40-47). The screen ran on the n1 pod (same host type), GPU0.
- Venvs, model, CUDA tree and compile caches staged on local NVMe (`$STAGE`); the persistent copies on NFS.

## RTX 5090 (sm_120): not run

torch 2.9.1+cu128 carries sm_120. Whether sgl-kernel 0.3.19 carries sm_120 code for the kernels this load uses was not
checked. The gain comes from removed CPU waits per decode step, so its size depends on the step time and the CPU, and
would need its own calibration, after the 5090 host's driver-hang issue is ruled out.
