# Reference solution: SGLang #16521

- Upstream PR: https://github.com/sgl-project/sglang/pull/16521, merged 2026-02-10.
- Fix (merge commit): `2bfab1bb67460af8b125d94ca4c5ab0040c12cd2`.
- Parent (first parent, the task's baseline): `4262f5259b94c5c08779efff5017c0a6235bcb5f`.

## What it changes

One file, 2 lines: `python/sglang/srt/mem_cache/radix_cache.py`.

When a request finishes, `RadixCache.cache_finished_req` inserts the request's tokens into the prefix tree. The parent
keys that entry by `req.fill_ids`, which at that point holds only the prompt, so the KV computed while decoding the
answer is freed. The next turn, which resends the prompt and the answer, has to prefill the answer again. The fix keys
the entry by `token_ids` = prompt + output (up to the committed KV length), so the next turn finds the answer in the
cache. Per turn-2 request on this load, `meta_info.cached_tokens` goes from 256 (the prompt) to 512 (prompt +
answer).

## How the package builds the fixed arm

- SGLang has no per-commit wheels. Each arm's package is built from the commit's source tarball
  (`codeload.github.com/sgl-project/sglang/tar.gz/<commit>`), of which only `python/` is extracted and built.
  Every extracted file must hash as in the authors' record (`package/env/<role>_src.sha256`).
- Dependencies come from one hashed lock (`package/env/deps.lock`, 167 packages), resolved from PyPI as of the merge
  date + 2 days: torch 2.9.1 (cu128), sgl-kernel 0.3.21, flashinfer 0.6.2, triton 3.5.1, transformers 4.57.1.
- The fix venv is the parent's exact freeze with only the sglang package swapped. Between the two venvs only
  `sglang/srt/mem_cache/radix_cache.py` and the generated `sglang/_version.py` differ
  (`package/env/files_differing.txt`).
- `package/setup/setup_lock.sh` builds both venvs and checks the freeze and every installed sglang file against the
  authors' build. `package/harness/run.sh <label> score` (candidate `fix` by default) scores this arm.

## Result

REF **3.827** on RTX 4090 (8 jobs, 3.794-3.871); 3.524 on RTX 5090 (3 jobs). The fix passes every check, with a
teacher-forced max gap of 0.125 nats. See `../README.md` and `../results/summary.json`.
