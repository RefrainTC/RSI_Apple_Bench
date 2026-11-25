# Reference solution: vLLM #13365

- Upstream PR: https://github.com/vllm-project/vllm/pull/13365 ("[V1][Spec Decode] Optimize N-gram matching with
  Numba"), merged 2025-02-18.
- Fix (merge commit): `4c822298981a8f7521492075ff72659985fc4c3f` (vLLM 0.7.3.dev216).
- Parent (first parent, the task's baseline): `c8d70e2437feecdb3762ce17298df33439ae1bd1` (0.7.3.dev215, V1).

## What it changes

3 files, 127 lines. `vllm/v1/spec_decode/ngram_proposer.py`: the KMP search for the first earlier occurrence of the
last n tokens moves from a Python loop over numpy scalars into two `@numba.jit(nopython=True)` functions.
`vllm/v1/worker/gpu_model_runner.py`: `GPUModelRunner.__init__` compiles them once, on `np.zeros(1024, int32)`. The PR
also adds `numba==0.60.0` to the requirements. Proposals are unchanged, so outputs and acceptances are bit-identical.

## How the package builds the fixed arm

- Both arms are vLLM's per-commit wheels from wheels.vllm.ai (URLs and sha256 in `package/ENV.md`), installed into
  `13365_parent` and `13365_fix` from hashed locks. The wheels differ in `ngram_proposer.py`, `gpu_model_runner.py`
  and the compiled `_C` / `_moe_C` libraries.
- The human fix imports numba, which the parent venv lacks, so it could not run as an overlay there. The task's venv
  **`13365_base` = parent + numba 0.60.0 / llvmlite 0.43.0 copied from the fix venv**; both arms run on it (the parent
  never imports numba). `package/setup/setup_venvs.sh` builds all three and checks freezes and `vllm` file hashes.
- Calibration derives the oracle stage by comparing the `.py` files of the parent and fix venvs; the fix arm is the
  base venv plus that overlay. A fix-venv vs parent-venv pair without overlays gave 1.0915, the same as the overlay REF.
- `ops_iso13365/score_fix.sh` makes `tasks/iso13365/fixws` (pristine workspace + the fix) and scores it through the
  full gate, whose numba rule allows the CPU JIT for this task only.

## Result

REF **1.0934** on RTX 4090 (7 unflagged jobs on two servers, 1.0916-1.0961, band ±0.5 %). See `../README.md` and
`../results/summary.json`.
