# Environment spec for vLLM #35442 (speculative decoding on a hybrid model: non-blocking copy of the accepted-token counts)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/35442, merged 2026-03-03 |
| Fix (merge commit) | `ad9d09e2b8a601b50d07c76fb8736c2bbda2d6fb` (vLLM 0.16.1rc1.dev177) |
| Parent (first parent) | `4beebfd14650b1c6a687e7ab496d501423a0e50d` (vLLM 0.16.1rc1.dev176) |
| Changed by the PR | `vllm/v1/worker/gpu_model_runner.py` only. After each speculative-decoding step, the model runner counts how many draft tokens each request accepted. The parent copies these counts to the CPU with a blocking `.cpu().numpy()` and stores them one by one in a Python loop. That stalls the CPU until the GPU has finished the step, which undoes the overlap of async scheduling. The fix keeps the counts in a GPU buffer and copies them to the pinned CPU buffer with `copy_(non_blocking=True)`. The blocking path stays only for the mamba "align" cache mode, which this load does not use. The CPU copy is uploaded back to the GPU at the next step's input preparation. |

## vLLM wheels

Per-commit wheels from `wheels.vllm.ai` (URL and sha256 in `env/build.env`; the authors' records are
`env/<role>_SOURCE.txt`). Both link CUDA 12 (`libcudart.so.12`). Nothing is compiled.

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 178 pinned packages with sha256 hashes, the same for parent and fix. Main pins: torch 2.10.0 (PyPI;
  CUDA 12.8 build), torchvision 0.25.0, torchaudio 2.10.0, triton 3.6.0, flashinfer-python 0.6.4, transformers 4.57.6,
  xgrammar 0.1.29, numpy 2.2.6 (all versions: `env/parent_freeze.txt`).
- How the lock was made: the authors (batch-5 screen on main) installed the parent wheel with uv `--exclude-newer
  2026-03-05` (merge date + 2 days). The fix venv is a copy of the parent's with only the vllm wheel swapped (`--no-deps`).
  Both wheels declare the same requirements. The freezes are `env/{parent,fix}_freeze.txt`; they differ only in the vllm
  line (`env/freeze.diff`). On 2026-11-25 the parent freeze minus the vllm line was compiled with `uv pip compile
  --no-deps --generate-hashes --exclude-newer 2026-03-05`.
- Checks done by `setup/setup_lock.sh`:
  - `uv pip freeze` equals the authors' freeze (apart from the vllm line's path);
  - all installed `vllm/` files (non-`.pyc`) hash exactly as `env/{parent,fix}_vllm.sha256`;
  - `import vllm, torch` works.

  Files that differ between parent and fix: `env/files_differing.txt`.

## Model

`Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (the snapshot batch 4 used, the same as
`packages/vllm37338`), whole repository (~9 GB), bf16. Qwen3.5-4B is a hybrid model: most layers are Gated DeltaNet
(linear attention), every fourth a full-attention layer. It has one MTP layer (`mtp_num_hidden_layers` 1), which vLLM
uses as the draft model for `speculative_config.method = "qwen3_5_mtp"` (logged as deprecated and mapped to `mtp`).
LFS blobs (sha256) are listed in `env/build.env` (`MODEL_LFS`).

## Engine settings of the load

`LLM(model, gpu_memory_utilization=0.80, seed=0, max_model_len=4096, enable_prefix_caching=False,
speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1})`. vLLM logs "Asynchronous scheduling is
enabled". Prefix caching is off, so the mamba cache mode is "none" (not "align"), and the fix's non-blocking path is taken.
Compile caches (Triton, Inductor, vLLM's torch.compile cache) live under `$STAGE/cache/<pr>/<role>/<host>-g<gpu>` and
are kept between runs of the same role (the scored region is the steady decode after start-up).

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4), main pod of 4090-server (Threadripper PRO 5975WX, 4 CCDs). One
  CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU0 0-7,32-39, GPU1 8-15,40-47, GPU2 16-23,48-55, GPU3
  24-31,56-63.
- Venvs and model staged on local NVMe (`$STAGE`); the persistent copies on NFS.

## RTX 5090 (sm_120): not run

torch 2.10.0's CUDA 12.8 build carries sm_120, and Triton generates sm_120 code. A run there needs:
- a driver for CUDA >= 12.8;
- a check that this vLLM wheel's own kernels (`_C`, flash-attention) carry sm_120 code (not checked here);
- its own calibration, since the gain depends on the CPU/GPU overlap;
- the 5090 host's driver-hang issue ruled out first.
