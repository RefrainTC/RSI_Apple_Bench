# Reference solution: vLLM #27418

- PR: https://github.com/vllm-project/vllm/pull/27418, merged 2025-10-24 (upstream issue vllm#27406; torch issue
  pytorch#166122).
- Parent (the task's baseline): `88d3141ec63008d8dff63311359355ebbe98ad64` (vLLM 0.11.1rc3.dev49).
- Fix (the merge commit): `42efe609ba75eb1b0bc06ae635778b2bc0aa4e7a` (vLLM 0.11.1rc3.dev50).

## What the fix changes

Six files, all in `vllm/model_executor/models/`: `qwen2_vl.py`, `qwen2_5_vl.py`, `qwen3_vl.py`,
`qwen3_omni_moe_thinker.py`, `glm4_1v.py` and `vision.py` (plus the generated `vllm/_version.py`).

The patch embedding of the Qwen2-VL family's vision tower was an `nn.Conv3d` whose kernel equals its stride, so every
output is one dot product over a 3 x 2 x 14 x 14 patch. The fix makes it a Linear on the flattened patch and reshapes
the checkpoint's Conv3d weight at load time. Under torch 2.9.0 with cuDNN 9.10, the bf16 Conv3d does not reach cuDNN
and runs the native `slow_conv_dilated3d` path (`vol2col` + GEMM): on an RTX 4090 ~22 ms per 1,024 patches, against
0.07 ms for the Linear (290x; 900x at 32k patches). Inside the parent's engine the slow path took 0.74 s of a 2.6 s rep
on the screen's first load.

The Linear computes the same products and sums them in a different order, so image outputs are close to the parent's
but not bit-identical; text-only outputs are identical. That is why this task uses a tolerance gate.

## How the package builds the fixed arm

Nothing of vLLM is compiled. Both venvs come from vLLM's per-commit wheels on `wheels.vllm.ai` (URLs and sha256 in
`package/env/build.env`). The parent venv is the parent wheel plus a hashed lock of its dependencies, resolved as of
the merge date + 2 days (`package/env/deps.lock`). Two packages are installed outside the lock:
- flashinfer-python 0.4.1, a hash-pinned sdist built without build isolation, because its build requirements pin a
  pre-release of apache-tvm-ffi that was deleted from PyPI (`apache-tvm-ffi==0.1.0` is used instead, in both arms);
- triton-kernels from git, pinned by commit.

The fix venv is a copy of the parent's with only the vllm wheel swapped (`--no-deps`). `setup/setup_lock.sh` checks
both freezes against the authors' and every vllm file against the authors' hashes; `package/env/files_differing.txt`
lists the 7 files that differ.

To score the reference solution: `bash harness/run.sh s1_g0 score` (the candidate defaults to `fix`).
