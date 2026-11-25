# Environment spec for vLLM #27418 (Qwen2-VL vision tower: patch embedding Conv3d -> Linear)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/27418, merged 2025-10-24 (issue vllm#27406; pytorch#166122) |
| Fix (merge commit) | `42efe609ba75eb1b0bc06ae635778b2bc0aa4e7a` (vLLM 0.11.1rc3.dev50) |
| Parent (first parent) | `88d3141ec63008d8dff63311359355ebbe98ad64` (vLLM 0.11.1rc3.dev49) |
| Changed by the PR | The patch embedding of the Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3-Omni and GLM-4.1V vision towers (6 files under `vllm/model_executor/models/`; list in `env/files_differing.txt`). The parent uses an `nn.Conv3d` whose kernel equals its stride, so each output is one dot product over a 3 x 2 x 14 x 14 patch. The fix replaces it with a `ReplicatedLinear` on the flattened patch, and reshapes the checkpoint's Conv3d weight at load time (`conv3d_to_linear_weight`). Under torch 2.9.0 (cuDNN 9.10) the bf16 Conv3d does not use cuDNN; it runs torch's native `slow_conv_dilated3d` (`vol2col` + GEMM). On an RTX 4090 that takes ~22 ms per 1,024 patches, against 0.07 ms for the Linear. |

## vLLM wheels

Per-commit wheels from `wheels.vllm.ai` (URL and sha256 in `env/build.env`; the authors' records are
`env/<role>_SOURCE.txt`). Both link CUDA 12 (`libcudart.so.12`). Nothing is compiled.

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 156 pinned packages with sha256 hashes, the same for parent and fix. Main pins: torch 2.9.0
  (CUDA 12.8 build, cuDNN 9.10.2.21), triton 3.5.0, transformers 4.57.1, apache-tvm-ffi 0.1.0, numpy 2.2.6,
  pillow 12.0.0. All versions: `env/parent_freeze.txt`.
- Two packages are installed outside the hashed lock, the same way the authors' screen build got them:
  - **flashinfer-python 0.4.1.** The parent wheel pins it. It exists only as an sdist (hash-pinned in `env/build.env`).
    Its `build-system.requires` pins `apache-tvm-ffi==0.1.0b15`, a pre-release that is no longer on PyPI. It is therefore
    built **without build isolation**, on top of the lock's `apache-tvm-ffi==0.1.0` (the oldest remaining release,
    2025-10-21, inside the cutoff), `setuptools`, `packaging`. The build is pure Python. The greedy load does not call
    FlashInfer's sampler.
  - **triton-kernels.** A git dependency of this vLLM, pinned by commit `c3c476f3` (`env/build.env`). It needs `git` on
    the host. It is used by MoE models only, not by Qwen2-VL.
- How the lock was made: the authors (batch-5 screen on main) installed the parent wheel with uv
  `--exclude-newer 2025-10-26` (merge date + 2 days), the override `apache-tvm-ffi==0.1.0` and the no-build-isolation
  step above. The fix venv is a copy of the parent's with only the vllm wheel swapped (`--no-deps`). The freezes are
  `env/{parent,fix}_freeze.txt` and differ only in the vllm line (`env/freeze.diff`). On 2026-11-25 the parent freeze,
  without vllm, flashinfer-python and triton-kernels, was compiled with `uv pip compile --no-deps --generate-hashes
  --exclude-newer 2025-10-26`.
- Checks done by `setup/setup_lock.sh`:
  - `uv pip freeze` equals the authors' freeze (apart from the vllm line's path and the flashinfer line's sdist URL);
  - all installed `vllm/` files (non-`.pyc`) hash exactly as `env/{parent,fix}_vllm.sha256`;
  - `import vllm, torch` works.

## Model

`Qwen/Qwen2-VL-2B-Instruct`, revision `1860c46f12b7449d0f5b426a7474ab715342bc65`, whole repository (~4.4 GB), bf16.
LFS blobs (sha256) are in `env/build.env` (`MODEL_LFS`). The vision tower has 32 blocks of width 1280; its patch embedding
reads 3 channels x 2 temporal frames x 14 x 14 pixels. A still image is repeated over the two frames. 2 x 2 patches are
merged into one LLM token, so a 1344 x 1344 image = 96 x 96 = 9,216 patches = 2,304 image tokens.

## Engine settings of the load

`LLM(model, gpu_memory_utilization=0.80, seed=0, max_model_len=4096, max_num_seqs=32, limit_mm_per_prompt={"image": 1},
mm_processor_kwargs={"min_pixels": 1806336, "max_pixels": 1806336}, mm_encoder_attn_backend="TORCH_SDPA")`.
- `mm_encoder_attn_backend`: on sm_89 this vLLM picks xFormers for the Qwen2-VL vision attention (FlashAttention does not
  take its head size). xFormers is not a dependency of the wheel, so the first image fails without this setting. SDPA is
  set in both arms.
- The teacher-forced re-read runs the parent with `gpu_memory_utilization` 0.60 and prefix caching off (`loads/tf.env`).
- Compile caches are kept per role, host and GPU under `$STAGE/cache/<pr>/`.

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4), main pod of 4090-server (Threadripper PRO 5975WX, 4 CCDs). One
  CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU0 0-7,32-39, GPU1 8-15,40-47, GPU2 16-23,48-55, GPU3
  24-31,56-63.

## RTX 5090 (sm_120): not run

torch 2.9.0+cu128 carries sm_120. Whether the slow Conv3d path is also taken there (it depends on torch / cuDNN, not on the
GPU) and how large the gain is would need a run, a new calibration, and the 5090 host's driver-hang issue ruled out first.
