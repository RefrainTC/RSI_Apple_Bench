# Reference solution: SGLang #20972

- Upstream PR: https://github.com/sgl-project/sglang/pull/20972, merged 2026-03-27.
- Fix (merge commit): `a27651d5e0d138d2a5ddc529ee95317fd57328fa`.
- Parent (first parent, the task's baseline): `e2b8463c80367730db57d5c38c8835ab326ab468`.

## What it changes

Two files: `python/sglang/srt/layers/sampler.py` and
`python/sglang/srt/managers/scheduler_output_processor_mixin.py`.

With `return_logprob` and `top_logprobs_num` > 0, the parent's sampler computes the top-k logprobs and calls
`.tolist()` on them right away (`get_top_logprobs`), and does the same for token-id logprobs. That is a blocking
GPU-to-CPU copy in the forward thread on every decode step, which also stops the overlap scheduler from preparing step
n+1 while the GPU runs step n. The fix keeps these tensors on the GPU (`no_copy_to_cpu=True`) and converts them in the
result processing after the step's copy-done event, for prefill and decode. No arithmetic changes: outputs and
logprobs are bit-identical to the parent's.

## How the package builds the fixed arm

- Each arm's SGLang is built from the commit's source tarball (`python/` only); every extracted file must hash as in
  the authors' record (`package/env/<role>_src.sha256`).
- Dependencies come from one hashed lock (`package/env/deps.lock`), resolved from PyPI as of the merge date + 2 days:
  torch 2.9.1 (cu128), sglang-kernel 0.4.0, flashinfer 0.6.6, triton 3.5.1, transformers 5.3.0.
- The fix venv is the parent's exact freeze with only the sglang package swapped. Only the two files above differ
  (`package/env/files_differing.txt`).
- `package/setup/setup_lock.sh` builds both venvs and checks them against the authors' build.
  `package/harness/run.sh <label> score` (candidate `fix` by default) scores this arm.

## Result

REF **1.142** on RTX 4090 (7 jobs, 1.137-1.148); 1.226 on RTX 5090 (3 jobs). The fix passes every check, bit-exact in
every rep. See `../README.md` and `../results/summary.json`.
