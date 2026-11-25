# Reference solution: vLLM #12380

- Upstream PR: https://github.com/vllm-project/vllm/pull/12380 ("[perf] fix perf regression from #12253"), merged
  2025-01-24.
- Fix (merge commit): `6dd94dbe94c1820a1e224cba65efcf0befa97995` (vLLM 0.6.6.post2.dev365).
- Parent (first parent, the task's baseline): `0e74d797ce8618fdb685126e0ff8576fb966e6ad` (0.6.6.post2.dev364, V0).

## What it changes

One file, 4 lines: `vllm/worker/model_runner.py`. PR #12253 made `ModelInputForGPUBuilder` persistent across steps
but left `self.decode_only = True` in `__init__`; after the first prefill (the start-up profile run) it stays False, so
no later batch is eligible for CUDA-graph replay. The fix resets `decode_only` in `prepare()`, once per batch. Decode
steps then replay the captured graph (100 % of decode steps on this load, against 0 % in the parent).

## How the package builds the fixed arm

- Both arms are vLLM's per-commit wheels from wheels.vllm.ai (URLs and sha256 in `package/ENV.md`), installed into two
  venvs (`12380_parent`, `12380_fix`) from hashed locks with `--no-deps`. They differ only in `vllm/_C.abi3.so`,
  `vllm/_version.py` and `vllm/worker/model_runner.py`. `package/setup/setup_venvs.sh` checks the freeze and every
  installed `vllm` file against the authors' records.
- Calibration derives the oracle stage by comparing the `.py` files of the two venvs. The fix arm runs the parent venv
  with that overlay, exactly as a candidate would; a venv-equivalence pair (fix venv vs parent venv, no overlay) gave
  1.5334 in the authors' package test.
- `harness_iso12380/mk_fix_ws.sh tasks/iso12380 <dir>` copies the pristine workspace and applies the fix to `repo/`;
  that workspace goes through the normal gate and `score.sh`.

## Result

REF **1.507** on RTX 4090 (8 jobs on two servers, 1.497-1.519, band ±1.1 %). Outputs are bit-identical to the parent's
(80/80 timed requests, 25/25 correctness prompts). See `../README.md` and `../results/summary.json`.
