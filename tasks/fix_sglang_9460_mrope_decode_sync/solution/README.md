# Reference solution: SGLang #9460

- Upstream PR: https://github.com/sgl-project/sglang/pull/9460, merged 2025-09-03.
- Fix (merge commit): `4ed9053ecffb910c87d6c0db4ef64a7c54aa099b`.
- Parent (first parent, the task's baseline): `5e19b159b02ae1b519ce6c3a0d1e7ff04ab9fcd5`.

## What it changes

Two files:
- `python/sglang/srt/model_executor/forward_batch_info.py`: the decode branch of
  `ForwardBatch._compute_mrope_positions`.
- `python/sglang/srt/layers/rotary_embedding.py`: loses the now unused helper.

In the parent's decode branch, each request's next M-RoPE position is built on the CPU: `int(self.seq_lens[i])` twice
per request (a blocking GPU-to-CPU read each, since `seq_lens` lives on the GPU), a Python `range`, a new CPU tensor
and its copy to the GPU. At batch 64 that is ~128 syncs and 64 small copies per decode step, and the syncs also stall
the overlap scheduler. The fix computes `delta + seq_lens[i] - 1` on the GPU (`torch.full` for text-only requests),
with a non-blocking copy of the delta.

The text-only branch most likely still syncs once per request: it passes the 0-d GPU tensor `seq_lens[i] - 1` to
`torch.full`, which takes a Python number. That is why the text-only transfer load gains less (1.10x). Not profiled.
A later SGLang PR, #14956, removes a decode-path `.item()` on text-only requests; it is the task
`fix_sglang_14956_mrope_text_sync`.

## How the package builds the fixed arm

- Each arm's SGLang is built from the commit's source tarball (`python/` only, setuptools backend); every extracted
  file must hash as in the authors' record (`package/env/<role>_src.sha256`).
- Dependencies come from one hashed lock (`package/env/deps.lock`), as of the merge date + 2 days: torch 2.8.0
  (cu128), sgl-kernel 0.3.8, triton 3.4.0, transformers 4.56.0.
- The fix venv is the parent's freeze with only the sglang package swapped (`--no-deps`). Only the two files above
  differ (`package/env/files_differing.txt`). No CUDA toolkit is needed.
- `package/setup/setup_lock.sh` builds both venvs and checks them against the authors' build.
  `package/harness/run.sh <label> score` (candidate `fix` by default) scores this arm.

## Result

REF **1.203** on RTX 4090 (8 jobs, 1.191-1.212). The fix passes every check; its batch-1 correctness pass is
bit-identical to the parent's. Not runnable on RTX 5090. See `../README.md` and `../results/summary.json`.
