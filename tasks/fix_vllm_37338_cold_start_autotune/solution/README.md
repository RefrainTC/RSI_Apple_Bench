# Reference solution: vLLM #37338

- Upstream PR: https://github.com/vllm-project/vllm/pull/37338, merged 2026-03-23.
- Fix (merge commit): `a16133a0f12e8e406206be1bc8041727d9f1ea7d` (vllm 0.18.1rc1.dev38).
- Parent (first parent, the task's baseline): `54ab804e87dd534c0a8cb8ad8eec7ad5d75775ae` (vllm 0.18.1rc1.dev37).

## What it changes

The substantive change is in `vllm/model_executor/models/qwen3_next.py`: the GDN warm-up now builds its inputs the way
a real prefill does, with `fused_gdn_gating` (fp32 `g` and `beta`) and int32 `cu_seqlens`. The parent built `g` and
`beta` with `torch.randn` in bf16 and `cu_seqlens` as int64. Triton's autotune cache is keyed by argument dtypes, so
the parent's warm-up tuned kernel variants that no real batch uses, and the first real prefill autotuned and compiled
the right ones (~46 s and ~100 extra kernels on a 4090).

Nine more files change only type hints, all under `vllm/model_executor/layers/fla/ops/`: `chunk_delta_h.py`,
`chunk_o.py`, `chunk.py`, `chunk_scaled_dot_kkt.py`, `fused_recurrent.py`, `fused_sigmoid_gating.py`, `index.py`,
`kda.py`, `wy_fast.py`. `vllm/_version.py` also differs.

The cost is removed, not only moved: with the fix, start-up is 0-5 s longer (the warm-up now compiles the kernels a
real prefill uses), and the first batch is ~47 s shorter.

## How the package builds the fixed arm

- vLLM publishes per-commit wheels. Each arm installs its commit's wheel from wheels.vllm.ai (CUDA 12 variant,
  sha256-pinned in `package/env/build.env`).
- Dependencies come from one hashed lock of the authors' freeze (`package/env/deps.lock`), as of the merge date + 2
  days: torch 2.10.0 (cu128), triton 3.6.0, flashinfer-python 0.6.6, transformers 4.57.6.
- The two venvs differ only in the vllm wheel: the 10 Python files above plus `_version.py`
  (`package/env/files_differing.txt`). Bytecode is pre-compiled, because the harness checks that runs do not write
  into the venv. No CUDA toolkit is needed.
- `package/setup/setup_lock.sh` builds both venvs and checks every vllm file against the authors' build.
  `package/harness/run.sh <label> score` (candidate `fix` by default) scores this arm.

## Result

REF **1.29** on RTX 4090 (10 jobs, 1.269-1.297); 1.178 on the RTX 5090 workstation (3 jobs). The fix passes every
check (counts, settings, teacher-forced, audit, venv, cold witness, steady, after). See `../README.md` and
`../results/summary.json`.
