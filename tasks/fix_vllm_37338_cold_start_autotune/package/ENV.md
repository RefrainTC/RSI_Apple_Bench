# Environment spec for vLLM #37338 (Qwen3.5 GDN warm-up with the dtypes of a real prefill)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/37338, merged 2026-03-23 |
| Fix (merge commit) | `a16133a0f12e8e406206be1bc8041727d9f1ea7d` (wheel `vllm-0.18.1rc1.dev38+ga16133a0f`) |
| Parent (first parent) | `54ab804e87dd534c0a8cb8ad8eec7ad5d75775ae` (wheel `vllm-0.18.1rc1.dev37+g54ab804e8`) |
| Changed by the PR | `vllm/model_executor/models/qwen3_next.py`: the GDN prefill warm-up built `g` and `beta` with `torch.randn` in the model dtype (bf16) and `cu_seqlens` as int64, while a real prefill passes `g` / `beta` from `fused_gdn_gating` (fp32) and int32 `cu_seqlens`; Triton's autotuner keys on argument dtypes, so the warm-up tuned kernels a real batch never uses and the first real prefill autotuned and compiled ~100 more kernel variants (~46 s on a 4090). The fix builds `g` / `beta` with `fused_gdn_gating` and `cu_seqlens` as int32 (and `output_final_state=True`). The other 9 changed files under `vllm/model_executor/layers/fla/ops/` only change type hints and docstrings (`torch.LongTensor` -> `torch.Tensor`). |

## vLLM wheels

Per-commit wheels from `wheels.vllm.ai` (URLs and sha256 in `env/build.env`; the CUDA 12 variant, `libcudart.so.12`).
The two wheels contain the same compiled libraries; their `vllm/` trees differ in the 10 Python files above and
`vllm/_version.py` (`env/files_differing.txt`).

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 174 pinned packages with sha256 hashes, the same for parent and fix. Main pins: torch 2.10.0 (cu128
  wheels from PyPI), torchvision 0.25.0, triton 3.6.0, flashinfer-python 0.6.6, transformers 4.57.6, xgrammar 0.1.32,
  huggingface-hub 0.36.2, numpy 2.2.6, nvidia-cutlass-dsl 4.4.2, setuptools 82.0.1.
- How the lock was made: the authors (batch 4, `SCREEN_BATCH4.md`, candidate C19) installed the parent wheel with uv
  `--exclude-newer 2026-03-25` (merge date + 2 days) from PyPI; the fix venv was a copy of the parent venv with only the
  vllm wheel swapped (`--no-deps`), after checking that both wheels declare the same `Requires-Dist`. Their freezes are
  `env/{parent,fix}_freeze.txt` (they differ only in the vllm line). On 2026-11-24 the parent freeze minus the vllm line
  was compiled with `uv pip compile --no-deps --generate-hashes --exclude-newer 2026-03-25 --python-version 3.10
  --python-platform x86_64-unknown-linux-gnu`.
- `setup/setup_lock.sh` installs lock + wheel with `--no-deps --require-hashes`, pre-compiles all bytecode (the harness
  fails a candidate whose runs write into its venv, so the venv must already hold every `.pyc` a run would write), and
  checks: `uv pip freeze` equals the authors' freeze (vllm line aside), every installed `vllm/` file (non-`.pyc`) hashes
  as `env/{parent,fix}_vllm.sha256`, `import vllm, torch`.
- No CUDA toolkit is needed (Triton ships its own `ptxas`; this load does not JIT-compile FlashInfer modules).

One known install-order effect: the `flashinfer-python` 0.6.6 and `torch-c-dlpack-ext` 0.1.5 wheels both ship a top-level
`site-packages/build_backend.py` (a packaging slip in both); uv installs in parallel, so which copy remains differs
between builds (the empty-directory fix venv kept torch-c-dlpack-ext's, the other three venvs flashinfer's). Nothing
imports it at run time and `setup_lock.sh` does not check it; the audit lists it as a differing `.py` file (not a
finding).

## Model

`Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (the snapshot batch 4 used), whole repository
(8.8 GB), bf16, used text-only. LFS blobs (sha256): `cb544bd9...e188`, `26a93f06...4e61` (the two safetensors shards),
`5f9e4d49...1cb42` (tokenizer.json); full values in `env/build.env`. `get_model.sh` writes `refs/main` = that revision.

## Cold start: what the harness clears

Each timed run gets new, empty directories for `TRITON_CACHE_DIR`, `TRITON_HOME`, `TORCHINDUCTOR_CACHE_DIR`,
`VLLM_CACHE_ROOT` (torch.compile artifacts), `VLLM_CONFIG_ROOT`, `XDG_CACHE_HOME`, `CUDA_CACHE_PATH` (driver JIT cache),
`FLASHINFER_WORKSPACE_BASE`, `TORCH_HOME`, plus a fresh `HOME`, `TMPDIR` and working directory (`harness/job.sh`), all
deleted after the run. The model weights stay in the OS page cache between runs (an unscored warm-up run per job makes
that equal for both arms); the score is about compile and autotune state, not disk reads.

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4; the cu128 torch wheels run on it), main pod of 4090-server
  (Threadripper PRO 5975WX, 4 CCDs). One CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU0 0-7,32-39,
  GPU1 8-15,40-47, GPU2 16-23,48-55, GPU3 24-31,56-63. Start-up compiles on the CPU (torch.compile / Inductor, Triton),
  so the cold time depends on the 16 logical CPUs a job gets.

## RTX 5090 (sm_120): what a run there would need (not run)

Checked on CPU by reading the wheels' fatbin sections (2026-11-24, `c5y_work/pod/scripts/archcheck.sh`):
- torch 2.10.0+cu128 carries sm_120; Triton 3.6.0 generates sm_120 code (the GDN kernels this task is about are Triton).
- vLLM's `_C.abi3.so` and `_moe_C.abi3.so` carry sm_120 SASS. `vllm_flash_attn/_vllm_fa2_C.abi3.so` carries only sm_80
  SASS + sm_80 PTX, so on a 5090 FlashAttention 2 would run from PTX JIT-compiled by the driver at first use (which our
  harness's empty `CUDA_CACHE_PATH` makes part of every cold start, in both arms); `_vllm_fa3_C` (sm_90) and `_flashmla_C`
  (sm_90/100) are not used on sm_89/sm_120. FlashInfer 0.6.6 ships no cubins for this load and would JIT with nvcc if
  vLLM chose a FlashInfer backend there (no CUDA toolkit on the pods), so the attention backend vLLM picks on sm_120
  must be checked first.
- Also needed: a driver for CUDA >= 12.8 (R570 or newer), its own calibration (cold start is dominated by CPU-side
  compilation, so REF depends on the host CPU as much as on the GPU), and the 5090 host's driver-hang issue fixed or
  ruled out first.
