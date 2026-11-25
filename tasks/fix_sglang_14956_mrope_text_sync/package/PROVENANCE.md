# Provenance of package sgl14956

The task was screened in batch 5 (`raw/P2-port-4090/SCREEN_BATCH5.md`, candidate C48; screened on n1 with the kit `/home/coder/work/b5/`,
rebuilt on main with its copy `/home/coder/work/c5y/b5m/`). This package was written on
2026-11-25 for the full task. Files derived from the batch-5 kit are listed with the original's sha256 (n1's copy; first 12
hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring numbers in the
card were produced by these same files (copied to `/home/coder/work/c5y/pkg/sgl14956` on the main pod), with the venvs
copied from the screen's build on main (`/tmp/c5y/b5m/venvs`; their freezes and sgl files equal `env/`); the
empty-directory test (`packages/VERIFY_batch5.md`) built everything from this package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 63b8fadc8548 | documentation |
| `RUN.md` | - | 7316730e28e1 | documentation |
| `bench/run_bench.sh` | batch 4 `bench/run_b5.sh` (70db12a67111) | 2ab0ed836f94 | run_b5.sh + with_env.sh in one file, path from $ISO_ROOT (identical to packages/vllm37338/bench/run_bench.sh) |
| `bench/sgl_c5y.py` | batch 4 `bench/sgl_b5.py` (7788bbc79499) | 3f609e401bee | default (random-token) mode unchanged (same prompts, seeds, warm-up, digests, metric); IMG mode as the screen's (images written under the run's directory); added: ids.json, settings.json and the CORRECT (correctness pass) mode; LOGPROB / CLP / MT modes dropped; via packages/sgl27868/bench/sgl_c5y.py |
| `env.sh` | - | 815415d46e86 | packages/sgl27868/env.sh, unchanged |
| `env/build.env` | - | d90e106b580e | new: commits, URLs, model revision (data) |
| `env/cuda_headers.lock` | - | a89e5924e044 | packages/sgl27868/env/cuda_headers.lock, unchanged (the same CUDA 12.8 header wheel versions as this torch 2.9.1) |
| `env/cuda_tree.sha256` | - | 49c5d255f5ee | packages/sgl27868/env/cuda_tree.sha256, unchanged (batch 4's /tmp/b4/cuda; the screen's /tmp/b5/cuda was built by the same pinned recipe) |
| `env/deps.lock` | - | 842dcadc8bf0 | generated 2026-11-25 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff 2025-12-15, PyPI only) |
| `env/files_differing.txt` | - | 6625c80c3952 | authors' record (c5y/b5m/setup/C48_sgl_pkg_files.txt) |
| `env/fix_freeze.txt` | - | 7fa05c266f24 | authors' record: the rebuild on main, c5y/b5m/setup/C48_fix_freeze.txt (equals n1's b5/setup apart from the source path) |
| `env/fix_sgl.sha256` | - | ed9ea5e25099 | authors' record: c5y/b5m/setup/C48_fix_sgl.sha256 (equals n1's) |
| `env/fix_src.sha256` | - | 8cfcccc6fc27 | authors' record: c5y/b5m/setup/C48_fix_src.sha256 |
| `env/freeze.diff` | - | ac3e6352f921 | authors' record (diff of the two freezes) |
| `env/parent_freeze.txt` | - | 3d58584dc270 | authors' record: the rebuild on main, c5y/b5m/setup/C48_parent_freeze.txt (equals n1's b5/setup apart from the source path) |
| `env/parent_sgl.sha256` | - | 009228fd45e9 | authors' record: c5y/b5m/setup/C48_parent_sgl.sha256 (equals n1's) |
| `env/parent_src.sha256` | - | da9bfaa8a3f2 | authors' record: c5y/b5m/setup/C48_parent_src.sha256 |
| `env/pyproject.diff` | - | e3b0c44298fc | authors' record (c5y/b5m/setup/C48_pyproject.diff): empty, parent and fix pyproject.toml identical |
| `harness/check.py` | - | c7d09403aa21 | packages/sgl27868/harness/check.py (exact gate), docstring for this PR; empty ref_correct.json tolerated |
| `harness/gate.json` | - | e19db2e23384 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b5_job.sh` (27058cfaefb1) | 1975e9a638d4 | batch-4/5 guarded job (b5_job.sh = b4_job.sh), machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE (identical to packages/vllm37338/harness/job.sh) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | packages/sgl27868/harness/mkvariant.sh, unchanged |
| `harness/ref_correct.json` | - | b6764d90e95a | authors' reference correctness pass (from calibration) |
| `harness/run.sh` | - | 88fc500b5eaa | packages/sgl27868/harness/run.sh + SGLANG_DISABLE_CUDNN_CHECK=1 for every run |
| `loads/cal.env` | batch 4 `env/C48.env` (9b7b87c60e16) | 2f09cb0410c5 | the screen's load (n1); bookkeeping variables (ID, BENCH, JOB_ENV) replaced by the harness (CUDA_HOME and SGLANG_DISABLE_CUDNN_CHECK set by run.sh) |
| `loads/cor.env` | - | d50f74f5f3c2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b8.env` | - | c514fe9283c6 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_img.env` | - | 7395d630761d | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/fetch_cuda.sh` | - | e20f7c2f0ada | packages/sgl27868/setup/fetch_cuda.sh, unchanged |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/sgl27868/setup/get_model.sh, unchanged |
| `setup/setup_lock.sh` | - | d95af6e60816 | packages/sgl27868/setup/setup_lock.sh: PyPI only (no PyTorch index), no package removed after the lock |
| `setup/stage.sh` | - | f5fd13224a84 | packages/sgl27868/setup/stage.sh, unchanged |
| `site.env.example` | - | 2e033717e004 | packages/sgl27868/site.env.example, unchanged |
| `variants/apply.py` | - | 9f615015d22f | new: the broken / cheating variants |
