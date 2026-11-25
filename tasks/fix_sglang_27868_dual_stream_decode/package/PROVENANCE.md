# Provenance of package sgl27868

The task was screened and briefly calibrated in batch 4 (`raw/P2-port-4090/SCREEN_BATCH4.md`, candidate C56; second host
in `SCREEN_BATCH4_main.md`; scripts in `raw/P2-port-4090/screen_batch4/`, pod copy `/home/coder/work/b4/`). This package
was written on 2026-11-24/25 for the full task. Files derived from batch-4 scripts are listed with the original's sha256
(first 12 hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring
numbers in the card were produced by these same files (copied to `/home/coder/work/c5y/pkg/sgl27868` on the main pod),
with the venvs taken from batch 4's build (copied from main's `/tmp/b4`, itself a byte-identical copy of n1's; their
sgl files and freezes equal `env/`); the empty-directory test (`packages/VERIFY_batch4b.md`) built everything from this
package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 07b3137307cb | documentation |
| `RUN.md` | - | 1f560d70e057 | documentation |
| `bench/run_bench.sh` | batch 4 `bench/run_b4.sh` (76f7fc43e372) | 2ab0ed836f94 | run_b4.sh + with_env.sh in one file, path from $ISO_ROOT |
| `bench/sgl_c5y.py` | batch 4 `bench/sgl_b4.py` (669cc1b1576e) | d6beeb114b52 | default (random-token) mode unchanged (same prompts, seeds, digests, metric); added: ids.json, settings.json and the CORRECT (correctness pass) mode; other modes dropped |
| `env.sh` | - | 815415d46e86 | packages/sgl16521/env.sh with this package's defaults (STAGE /tmp/c5y/st, PORT 36000 + 500 x GPU): caller's variables win over site.env |
| `env/build.env` | - | 9d20ae1a3dd5 | new: commits, URLs, model revision (data) |
| `env/cuda_headers.lock` | - | a89e5924e044 | the five CUDA 12.8 header wheels with their hashes, copied from packages/sgl16521/env/deps.lock |
| `env/cuda_tree.sha256` | - | 49c5d255f5ee | authors' record: sha256 of every file of batch 4's /tmp/b4/cuda (identical to packages/sgl16521/env/cuda_tree.sha256) |
| `env/deps.lock` | - | f12c8139e383 | generated 2026-11-24 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff merge date + 2 days); per-package cutoff exceptions for the five override packages; jinja2 / markupsafe hashes from PyPI |
| `env/files_differing.txt` | - | ccf73e910f46 | authors' record (batch 4 setup/C56_sgl_pkg_files.txt) |
| `env/fix_freeze.txt` | - | e98c51f85ef4 | authors' record, copied unchanged from batch 4 setup/C56_fix_freeze.txt |
| `env/fix_sgl.sha256` | - | e87127e0f516 | authors' record, copied unchanged from batch 4 setup/C56_fix_sgl.sha256 |
| `env/fix_src.sha256` | - | 2701b3a5bfe0 | authors' record, copied unchanged from batch 4 setup/C56_fix_src.sha256 |
| `env/freeze.diff` | - | 0628546e129c | authors' record (batch 4 setup/C56_freeze.diff) |
| `env/override.txt` | - | 33911a2d5c0c | authors' record (batch 4 setup/C56_override.txt): the cu129 pins |
| `env/parent_freeze.txt` | - | 6f21932fd385 | authors' record, copied unchanged from batch 4 setup/C56_parent_freeze.txt |
| `env/parent_sgl.sha256` | - | f912e3b9477f | authors' record, copied unchanged from batch 4 setup/C56_parent_sgl.sha256 |
| `env/parent_src.sha256` | - | e4e0a0224e12 | authors' record, copied unchanged from batch 4 setup/C56_parent_src.sha256 |
| `env/pyproject.diff` | - | e3b0c44298fc | authors' record (batch 4 setup/C56_pyproject.diff): empty, parent and fix pyproject.toml identical |
| `harness/check.py` | - | 5381330f26e3 | new: timing, output checks, gate, verdict |
| `harness/gate.json` | - | ff811635cb12 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b4_job.sh` (aa2df9db2850) | 1975e9a638d4 | batch-4 guarded job, machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE (via packages/sgl16521/harness/job.sh, byte-identical) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | new: broken-variant venvs (hard-linked copies + Python edits) |
| `harness/ref_correct.json` | - | 80fcdfba7749 | reference correctness pass (from calibration) |
| `harness/run.sh` | - | ca0ed175e981 | new: job lists of calibration / scoring / variants / transfers, STOP and lock handling |
| `loads/cal.env` | batch 4 `env/C56_cal.env` (94d06228cb92) | 2c4d78484093 | same load; batch-4 bookkeeping variables (ID, BENCH, JOB_ENV) replaced by the harness (CUDA_HOME set by run.sh) |
| `loads/cor.env` | - | 66e5f5bc8a0d | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b1.env` | - | 2ca7fca50781 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b8.env` | - | f2eedc5656f8 | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/fetch_cuda.sh` | - | e20f7c2f0ada | packages/sgl16521/setup/fetch_cuda.sh with the header wheels taken from env/cuda_headers.lock instead of the venv (this venv has CUDA 12.9 math libraries; batch 4 used 12.8 headers) |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/sgl16521/setup/get_model.sh, unchanged: model download at the pinned revision, blob sha256 check, retries on HTTP 429 |
| `setup/setup_lock.sh` | - | 0116d718e762 | new (structure of packages/sgl16521/setup/setup_lock.sh): builds both venvs from the hashed lock and checks them against the authors' records |
| `setup/stage.sh` | - | f5fd13224a84 | packages/sgl16521/setup/stage.sh structure: copies interpreter, venvs, CUDA tree, model to $STAGE and relocates the venvs |
| `site.env.example` | - | 2e033717e004 | new |
| `variants/apply.py` | - | 12f17a41a3dd | new: the broken / cheating variants |
