# Reference solution: SGLang #17005

- PR: https://github.com/sgl-project/sglang/pull/17005, merged 2026-01-13.
- Parent (the task's baseline): `075c5a5789e77de393dc5490099a8f7d3ccd4d12`.
- Fix (the merge commit): `339915ce2b8c9c6cb73a4a52d3d084d8f66c967f`.

## What the fix changes

One file: `python/sglang/srt/managers/tokenizer_manager.py` (plus the generated `sglang/_version.py`).

A request with `return_logprob` receives its new positions from the detokenizer in a message every 50 tokens when it
does not stream. On every message, the parent's `add_logprob_to_meta_info` converts **all** positions so far into lists
of Python tuples: the produced token's logprob, plus the top-k pairs when asked. That is O(n) per message and O(n²) per
request, in the tokenizer manager's process. For a 2,048-token request that is ~41 messages, each converting everything
again.

The fix keeps the converted lists in the request's state (`ReqState`) and converts only the new positions, extending the
cached lists. What the engine computes, and what the request returns, do not change, so outputs and logprobs are
bit-identical.

On this task's load (192 requests x 2,048 tokens, top 20 logprobs) that saves ~12 s per rep, 1.40x. Even with the fix,
a rep with top-20 logprobs takes ~7 s longer than one without logprobs; that time is not in the conversion of the top
lists (the `notop` variant skips it and is no faster), and where it goes was not measured. So the human fix is not the
ceiling on this load.

## How the package builds the fixed arm

Each arm is built from SGLang's source tarball of its commit (`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`);
only `python/` is extracted and built with the backend its `pyproject.toml` declares. Every extracted file must hash as
in the authors' record (`package/env/<role>_src.sha256`, 1,660 files). Dependencies come from a hashed lock of the
parent's resolution as of the merge date + 2 days (2026-01-15), from PyPI, with no overrides (`package/env/deps.lock`).
The fix venv is a copy of the parent's with only the sglang package swapped. `setup/setup_lock.sh` checks the freezes and
every sglang file against the authors' hashes; only `sglang/srt/managers/tokenizer_manager.py` differs, apart from the
generated `_version.py` (`package/env/files_differing.txt`). A CUDA 12.8 tree is fetched for FlashInfer's JIT
(`setup/fetch_cuda.sh`).

To score the reference solution: `bash harness/run.sh s1_g0 score` (the candidate defaults to `fix`).
