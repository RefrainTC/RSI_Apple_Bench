# Reference solution: vLLM #35442

- PR: https://github.com/vllm-project/vllm/pull/35442, merged 2026-03-03.
- Parent (the task's baseline): `4beebfd14650b1c6a687e7ab496d501423a0e50d` (vLLM 0.16.1rc1.dev176).
- Fix (the merge commit): `ad9d09e2b8a601b50d07c76fb8736c2bbda2d6fb` (vLLM 0.16.1rc1.dev177).

## What the fix changes

One file: `vllm/v1/worker/gpu_model_runner.py` (plus the generated `vllm/_version.py`).

With speculative decoding on a hybrid model, after every step the model runner needs the number of draft tokens each
request accepted, to advance the Gated-DeltaNet / Mamba state correctly. The parent copies these counts to the CPU
with a blocking `.cpu().numpy()` followed by a Python loop over requests. The blocking copy makes the CPU wait for the
GPU to finish the step, so the next step cannot be prepared while the GPU works, even with async scheduling on.

The fix keeps the counts in a GPU buffer and copies them to pinned CPU memory with `copy_(non_blocking=True)`. The
blocking path stays only for the mamba "align" cache mode, which this load does not use. What is computed does not
change, so outputs are bit-identical to the parent's.

The upstream issue measured the copy at 6 ms -> 0.2 ms (the 6 ms being time spent waiting for the GPU) and an MTP
latency regression of +76 %. On this task's load the saving is about 0.25-0.5 ms per decode step, 1.037x per rep.

One caveat from our measurements: the fix reads the pinned copy without waiting for the copy to finish. On this load
every fix run was bit-identical to the parent, so no stale read showed.

## How the package builds the fixed arm

Nothing is compiled. Both venvs come from vLLM's per-commit wheels on `wheels.vllm.ai` (URLs and sha256 in
`package/env/build.env`). The parent venv is the parent wheel plus a hashed lock of its dependencies, resolved as of
the merge date + 2 days (`package/env/deps.lock`). The fix venv is a copy of the parent's with only the vllm wheel
swapped (`--no-deps`); both wheels declare the same requirements. `setup/setup_lock.sh` checks that both freezes equal
the authors' and that every vllm file hashes as in the authors' build; only `vllm/_version.py` and
`vllm/v1/worker/gpu_model_runner.py` differ between the two (`package/env/files_differing.txt`).

To score the reference solution: `bash harness/run.sh s1_g0 score` (the candidate defaults to `fix`).
