# Reference solution: SGLang #14956

- PR: https://github.com/sgl-project/sglang/pull/14956, merged 2025-12-13.
- Parent (the task's baseline): `a7a4b1755d1b6eb7b04925f8453647368c029487`.
- Fix (the merge commit): `69cfb17b9a5bfde2b6b150989159a2942ef21b89`.
- Related: SGLang #9460 (task `fix_sglang_9460_mrope_decode_sync`), which removed the same kind of read for image
  requests in the same function.

## What the fix changes

One file: `python/sglang/srt/model_executor/forward_batch_info.py`, function `_compute_mrope_positions`, decode branch.

For a request without an image (`mm_input is None`), the parent builds the request's 3 x 1 mrope position with
`torch.full((3, 1), self.seq_lens[i] - 1, device=cuda)`. `self.seq_lens` is a GPU tensor, so the fill value is a 0-d
CUDA tensor and `torch.full` reads it back to the host: one blocking device-to-host copy per text request per decode
step, inside the forward thread, which also breaks the overlap scheduler's CPU/GPU overlap.

The fix reads `self.seq_lens_cpu[i] - 1` instead (the same number, already kept on the CPU), and takes the batch size
from the CPU copy as well. Nothing waits for the GPU. The value is the same, so outputs are bit-identical.

On this task's load (64 text requests, 256 decode steps per rep) that removes 64 waits per step, about 1.9 ms per step,
1.31x per rep.

## How the package builds the fixed arm

Each arm is built from SGLang's source tarball of its commit (`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`);
only `python/` is extracted and built with the backend its `pyproject.toml` declares. Every extracted file must hash as
in the authors' record (`package/env/<role>_src.sha256`). Dependencies come from a hashed lock of the parent's
resolution as of the merge date + 2 days, from PyPI, with no overrides (`package/env/deps.lock`). The fix venv is a copy
of the parent's with only the sglang package rebuilt from the fix's source; `setup/setup_lock.sh` checks the freezes and
every sglang file against the authors' hashes. Only `sglang/srt/model_executor/forward_batch_info.py` differs
(`package/env/files_differing.txt`). A CUDA 12.8 tree is fetched for FlashInfer's JIT (`setup/fetch_cuda.sh`).

To score the reference solution: `bash harness/run.sh s1_g0 score` (the candidate defaults to `fix`).
