# Provenance of package sgl17005

The task was screened in batch 7 (`raw/P2-port-4090/SCREEN_BATCH7.md`, candidate M2; screened on main by c5y with the batch-7 kit
`/home/coder/work/c5y/b7m/`, a copy of `c5y/b6m/`; see `SCREEN_BATCH7.md`). This package was written on
2026-11-25 for the full task. Files derived from the batch-7 kit are listed with the original's sha256 (main's c5y/b7m copy; first 12
hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring numbers in the
card were produced by these same files (copied to `/home/coder/work/c5y/pkg/sgl17005` on the main pod), with the venvs
copied from the screen's build on main (`/tmp/c5y/b7m/venvs`; their freezes and sgl files equal `env/`); the
empty-directory test (`packages/VERIFY_batch7.md`) built everything from this package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 6b325faf2391 | documentation |
| `RUN.md` | - | f968fd29022a | documentation |
| `bench/run_bench.sh` | batch 4 `bench/run_b5.sh` (9d3a0bfd9ab0) | 2ab0ed836f94 | run_b5.sh + with_env.sh in one file, path from $ISO_ROOT (identical to packages/vllm37338/bench/run_bench.sh) |
| `bench/sgl_c5y.py` | batch 4 `bench/sgl_b5.py` (7e520571ae4f) | a0342c6b7023 | packages/sgl20503/bench/sgl_c5y.py + the kit's LOGPROB mode (return_logprob, top_logprobs_num, digest over ids, text, produced-token and top-k logprobs rounded to 1e-3), as in the kit's sgl_b5.py, which the M2 screen ran (same prompts, seeds, warm-up, digests, metric); COR_OL for the correctness pass length |
| `env.sh` | - | 815415d46e86 | packages/sgl20503/env.sh, unchanged |
| `env/build.env` | - | 9694ae934dab | new: commits, URLs, model revision (data) |
| `env/cuda_headers.lock` | - | a89e5924e044 | packages/sgl20503/env/cuda_headers.lock, unchanged |
| `env/cuda_tree.sha256` | - | 49c5d255f5ee | packages/sgl20503/env/cuda_tree.sha256, unchanged |
| `env/deps.lock` | - | 3f8d9d004811 | generated 2026-11-25 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff 2026-01-15, PyPI only) |
| `env/files_differing.txt` | - | 2875d653e698 | authors' record (c5y/b7m/setup/M2_sgl_pkg_files.txt) |
| `env/fix_freeze.txt` | - | e9a1a8bb479c | authors' record: batch-7 screen build on main, c5y/b7m/setup/M2_fix_freeze.txt |
| `env/fix_sgl.sha256` | - | 5ff8fbfce75b | authors' record: c5y/b7m/setup/M2_fix_sgl.sha256 |
| `env/fix_src.sha256` | - | 62c9500d0cb4 | authors' record: c5y/b7m/setup/M2_fix_src.sha256 |
| `env/freeze.diff` | - | 847f6fef95f9 | authors' record (diff of the two freezes) |
| `env/parent_freeze.txt` | - | 208ad07329bb | authors' record: batch-7 screen build on main, c5y/b7m/setup/M2_parent_freeze.txt |
| `env/parent_sgl.sha256` | - | fb12b788862f | authors' record: c5y/b7m/setup/M2_parent_sgl.sha256 |
| `env/parent_src.sha256` | - | 72a71cc3e7ff | authors' record: c5y/b7m/setup/M2_parent_src.sha256 |
| `env/pyproject.diff` | - | e3b0c44298fc | authors' record (c5y/b7m/setup/M2_pyproject.diff): empty, parent and fix pyproject.toml identical |
| `harness/check.py` | - | 339f4f0dfa4b | packages/sgl20503/harness/check.py (exact gate), docstring for this PR; the correctness pass also compares the top-5 lists |
| `harness/gate.json` | - | ca181c4e54d1 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b5_job.sh` (832d75642eb8) | 395b804fcfd5 | batch-4/5 guarded job (b5_job.sh = b4_job.sh), machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE; since 2026-11-25 it also requires R4_ROOT=$ISO_ROOT when it looks for its own processes (acc6: two copies with the same labels could kill each other's runs) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | packages/sgl20503/harness/mkvariant.sh, unchanged |
| `harness/ref_correct.json` | - | 28fe6ae00ea3 | authors' reference correctness pass (from calibration) |
| `harness/run.sh` | - | 88fc500b5eaa | packages/sgl20503/harness/run.sh, unchanged |
| `loads/cal.env` | batch 4 `env/M2.env` (558395bebbbd) | 285aa4c1b80e | the screen's load; bookkeeping variables (ID, BENCH, JOB_ENV) replaced by the harness (CUDA_HOME and SGLANG_DISABLE_CUDNN_CHECK set by run.sh); TRANSFERS added |
| `loads/cor.env` | - | df9c78fd4dfa | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_k5.env` | - | 27dd43b83a8a | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_nolp.env` | - | 670c974ab138 | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/fetch_cuda.sh` | - | e20f7c2f0ada | packages/sgl20503/setup/fetch_cuda.sh, unchanged |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/sgl20503/setup/get_model.sh, unchanged |
| `setup/setup_lock.sh` | - | d95af6e60816 | packages/sgl20503/setup/setup_lock.sh, unchanged |
| `setup/stage.sh` | - | f5fd13224a84 | packages/sgl20503/setup/stage.sh, unchanged |
| `site.env.example` | - | 2e033717e004 | packages/sgl20503/site.env.example, unchanged |
| `variants/apply.py` | - | 6123b938a48b | new: the broken / cheating variants |
