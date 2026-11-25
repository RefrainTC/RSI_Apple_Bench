# Reference solution: SGLang #6994

- Upstream PR: https://github.com/sgl-project/sglang/pull/6994 ("[Perf] Refactor LoRAManager to eliminate stream syncs
  and redundant computations"), merged 2025-06-11.
- Fix (merge commit): `021f76e4f49861b2e9ea9ccff06a46d577e3c548` (SGLang 0.4.7).
- Parent (first parent, the task's baseline): `777688b8929c877e4e28c2eac208d776abe4c3af`.

## What it changes

Two files: `python/sglang/srt/lora/lora_manager.py` and `python/sglang/srt/lora/mem_pool.py`. In
`LoRAManager.prepare_lora_batch` the parent writes the adapter index, rank and scaling of each request into GPU tensors
one by one, indexing with 0-d GPU tensors (a GPU->CPU sync per request), and the eager branch calls
`int(torch.max(seg_lens))`. The fix builds the three lists on the CPU, copies them once from pinned memory with
`non_blocking=True`, takes `max_len` from the CPU copy of the extend lengths, and fills the constant CUDA-graph
`seg_lens` / `seg_indptr` once at start-up. On this load that saves ~1.2 s per rep of 256 decode steps.

## How the package builds the fixed arm

- SGLang has no per-commit wheels. Each arm's package is built from the commit's source tarball
  (`codeload.github.com/sgl-project/sglang/tar.gz/<commit>`); only `python/` is extracted and built by its setuptools
  backend. Every extracted file must hash as in the authors' record (`package/env/6994_<role>_src.sha256`).
- Dependencies come from one hashed lock (`package/env/6994_deps.lock`, 133 packages, identical for both arms): torch
  2.7.1 (cu126), sgl-kernel 0.1.7, triton 3.3.1, xgrammar 0.1.19. flashinfer-python 0.2.6.post1 is left out (an sdist
  that needs nvcc); the task's backends do not import it.
- `package/b3/scripts/setup_lock.sh` builds `6994_parent` and `6994_fix` and checks the freeze and all 696 installed
  `sglang/` files against the authors' build. Only `lora_manager.py` and `mem_pool.py` differ between them.
- The fix arm is that whole venv (no overlay): `b3/scripts/b3_cal.sh` runs role `fix` from `$STAGE/b3/venvs/6994_fix`.

## Result

REF **1.474** on RTX 4090 (8 jobs on 4 cards and two servers, 1.465-1.485, band ±1.1 %); correctness pass bit-identical
to the parent's in every job. See `../README.md` and `../results/summary.json`.
