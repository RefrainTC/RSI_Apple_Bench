# Reference solution: vLLM #14930

- Upstream PR: https://github.com/vllm-project/vllm/pull/14930 ("[V1][Spec Decode] Optimize Rejection Sampler with
  Triton Kernels"), merged 2025-03-18.
- Fix (merge commit): `99abb8b650c66664cdc84d815b7f306f33bd9881` (vLLM 0.8.0rc3.dev22).
- Parent (first parent, the task's baseline): `3a1e6481586ed7f079275b5d5072a6e246af691e` (0.8.0rc3.dev21, V1).

## What it changes

7 non-test files, ~1100 lines. The V1 rejection sampler (`vllm/v1/sample/rejection_sampler.py`) is rewritten as
hand-written Triton kernels for the greedy and the random path. `SpecDecodeMetadata` (`vllm/v1/spec_decode/metadata.py`)
is built on the GPU by the model runner (`vllm/v1/worker/gpu_model_runner.py`), so there is no Python list of drafts,
no padded `[batch, k+1, vocab]` probability tensor and no one-hot draft probabilities. A `torch.compile`d softmax is
added. Other files between the two wheels: `vllm/envs.py`, `vllm/v1/outputs.py`, `vllm/v1/sample/ops/utils.py`,
`vllm/v1/spec_decode/utils.py`, plus `_C.abi3.so` and `_version.py`.

**This solution is outside the agent rules**: the kernel gate rejects its new Triton code (24 hits), and
`vllm/envs.py` is a locked file. The package therefore scores it from the oracle stage with the gate bypassed
(`STAGE:` mode of `score.sh`). A torch-only rewrite that passes the gate (`package/ops_iso14930/mk_alt.py`, variant V1:
flat argmax over `[num_tokens, vocab]` with CPU-side split points, sampled ids from one `.tolist()`, ~20 lines) reaches
1.044x.

## How the package builds the fixed arm

- Both arms are vLLM's per-commit wheels from wheels.vllm.ai (URLs and sha256 in `package/ENV.md`), installed into
  `14930_parent` and `14930_fix` from hashed locks with `--no-deps`, including the `xgrammar==0.1.17` override (vLLM
  pins 0.1.16, which PyPI no longer serves). `package/setup/setup_venvs.sh` checks freezes and `vllm` file hashes.
- Calibration derives the oracle stage by comparing the `.py` files of the two venvs; the fix arm is the parent venv
  plus that overlay. A venv-equivalence pair (fix venv vs parent venv) gave 1.0915 in the authors' package test.
- `bash ops_iso14930/score_job.sh $GPU fix1 tasks/iso14930/calib/oracle_stage` scores it.

## Result

REF **1.0909** on RTX 4090 (6 jobs, 1.0885-1.0939, band ±0.5 %); greedy outputs identical to the parent's in every
trusted job. See `../README.md` and `../results/summary.json`.
