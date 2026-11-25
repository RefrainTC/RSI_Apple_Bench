# Reference solution: vLLM #16436

- Upstream PR: https://github.com/vllm-project/vllm/pull/16436 ("[Core] Speed up decode by remove synchronizing
  operation in sampler"), merged 2025-04-21.
- Fix (merge commit): `299ebb62b269ce167eb1c71b5e39a1dc1f65ce1c` (vLLM 0.8.5.dev118).
- Parent (first parent, the task's baseline): `f728ab8e3578c22b42ed53e51b5e8ec35328d8b9` (0.8.5.dev117).

## What it changes

One file, 14 lines: `vllm/model_executor/layers/utils.py`. The repetition penalty was applied with boolean-mask
indexing, whose output size depends on the data, so the GPU had to report it to the CPU on every sampling step. The fix
computes the same penalty with `torch.where`, which needs no sync. Median TPOT drops from ~5.45 ms to ~5.00 ms on this
load.

## How the package builds the fixed arm

- Both arms are vLLM's per-commit wheels from wheels.vllm.ai (URLs and sha256 in `package/ENV.md`), installed into two
  venvs (`16436_parent`, `16436_fix`) from hashed locks with `--no-deps`, then byte-compiled (without the bytecode
  every run would write `.pyc` files into its venv copy and be flagged `venv_changed`). `package/setup/setup_venvs.sh`
  checks the freeze and every installed `vllm` file against the authors' records.
- Calibration derives the oracle stage by comparing the `.py` files of the two venvs; the fix arm runs the parent venv
  with that overlay, as a candidate would. `harness/mk_fix_ws.sh tasks/iso16436 <dir>` puts the fix into a copy of the
  pristine workspace, which goes through the normal gate and `harness/score.sh`.

## Result

REF **1.0855** on RTX 4090 (7 jobs, 1.0816-1.0957, band ±1.0 %). See `../README.md` and `../results/summary.json`.
