# Reference solution: SGLang #20503

- PR: https://github.com/sgl-project/sglang/pull/20503, merged 2026-03-18.
- Parent (the task's baseline): `20a23e3173b5b3bdb16f19451f55d448b3747fb0`.
- Fix (the merge commit): `a6c7bb54eb3aeb60df44c6087ab7c929376e2022`.

## What the fix changes

One file: `python/sglang/srt/managers/scheduler.py`, +2/-3 lines in `Scheduler.get_new_batch_prefill`.

After a prefill admission, the parent removes the admitted requests from the waiting queue with

```python
[x for x in self.waiting_queue if x not in set(can_run_list)]
```

Python evaluates `set(can_run_list)` again for each of the W waiting requests, so one admission costs O(W x R) for R
admitted requests, in the scheduler's event loop. The fix builds the set once. Which
requests are admitted does not change, so outputs are bit-identical.

On this task's load (16,384 requests, at most 256 running, ~64 admission waves per rep) that saves ~12 ms per wave,
~0.75 s per rep, 1.08x.

A shortcut that is **not** equivalent: dropping the first R entries of the queue (`waiting_queue[len(can_run_list):]`,
the `prefix` variant). It gives the same queue on this load, where admission is first-come-first-served and nothing is
skipped, but is wrong with priority scheduling, LoRA batches or hierarchical-cache prefetch. The gate cannot tell the
two apart.

## How the package builds the fixed arm

Each arm is built from SGLang's source tarball of its commit (`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`);
only `python/` is extracted and built with the backend its `pyproject.toml` declares (setuptools). Every extracted file
must hash as in the authors' record (`package/env/<role>_src.sha256`, 2,162 files); the two `pyproject.toml` files are
identical. Dependencies come from a hashed lock of 170 packages resolved for the parent as of the merge date + 2 days,
from PyPI, with no overrides (`package/env/deps.lock`). The fix venv is a copy of the parent's with only the sglang
package swapped (`--no-deps`). `setup/setup_lock.sh` checks the freezes and every sglang file against the authors'
hashes; only `sglang/srt/managers/scheduler.py` differs, apart from the generated `_version.py`
(`package/env/files_differing.txt`). A CUDA 12.8 tree is fetched for FlashInfer's JIT (`setup/fetch_cuda.sh`).

To score the reference solution: `bash harness/run.sh s1_g0 score` (the candidate defaults to `fix`).
