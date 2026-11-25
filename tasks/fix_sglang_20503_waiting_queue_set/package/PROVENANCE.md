# Provenance of package sgl20503

The task was screened in batch 6 (`raw/P2-port-4090/SCREEN_BATCH6.md`, candidate N2; screened on main by c5y with the batch-6 kit
`/home/coder/work/c5y/b6m/`, a copy of `c5y/b5m/`; see `SCREEN_BATCH6.md`). This package was written on
2026-11-25 for the full task. Files derived from the batch-6 kit are listed with the original's sha256 (main's c5y/b6m copy; first 12
hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring numbers in the
card were produced by these same files (copied to `/home/coder/work/c5y/pkg/sgl20503` on the main pod), with the venvs
copied from the screen's build on main (`/tmp/c5y/b6m/venvs`; their freezes and sgl files equal `env/`); the
empty-directory test (`packages/VERIFY_batch6.md`) built everything from this package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 6887172c5624 | documentation |
| `RUN.md` | - | a6458d845e8a | documentation |
| `bench/run_bench.sh` | batch 4 `bench/run_b5.sh` (76e552816138) | 2ab0ed836f94 | run_b5.sh + with_env.sh in one file, path from $ISO_ROOT (identical to packages/vllm37338/bench/run_bench.sh) |
| `bench/sgl_c5y.py` | batch 4 `bench/sgl_b5.py` (930a24d8b62d) | f21440c4422f | the C48 package's sgl_c5y.py with a new docstring; its default (random-token) mode equals the kit's sgl_b5.py default mode, which the N2 screen ran (same prompts, seeds, warm-up, digests, metric). The kit's hash is of its later version, which also has the N4 additions (LORA, SEQ, logprob saving), none of them active by default |
| `env.sh` | - | 815415d46e86 | packages/sgl14956/env.sh, unchanged |
| `env/build.env` | - | 9724de01f8c7 | new: commits, URLs, model revision (data) |
| `env/cuda_headers.lock` | - | a89e5924e044 | packages/sgl14956/env/cuda_headers.lock, unchanged |
| `env/cuda_tree.sha256` | - | 49c5d255f5ee | packages/sgl14956/env/cuda_tree.sha256, unchanged |
| `env/deps.lock` | - | 483b4bffca20 | generated 2026-11-25 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff 2026-03-20, PyPI only) |
| `env/files_differing.txt` | - | b3ba5082c838 | authors' record (c5y/b6m/setup/N2_sgl_pkg_files.txt) |
| `env/fix_freeze.txt` | - | 2c2e88a0d2b1 | authors' record: batch-6 screen build on main, c5y/b6m/setup/N2_fix_freeze.txt |
| `env/fix_sgl.sha256` | - | af7136500f1a | authors' record: c5y/b6m/setup/N2_fix_sgl.sha256 |
| `env/fix_src.sha256` | - | e929694765e0 | authors' record: c5y/b6m/setup/N2_fix_src.sha256 |
| `env/freeze.diff` | - | 9777119baeea | authors' record (diff of the two freezes) |
| `env/parent_freeze.txt` | - | 3c12a85bca6e | authors' record: batch-6 screen build on main, c5y/b6m/setup/N2_parent_freeze.txt |
| `env/parent_sgl.sha256` | - | 1edfbf6635d2 | authors' record: c5y/b6m/setup/N2_parent_sgl.sha256 |
| `env/parent_src.sha256` | - | 860cffaf2745 | authors' record: c5y/b6m/setup/N2_parent_src.sha256 |
| `env/pyproject.diff` | - | e3b0c44298fc | authors' record (c5y/b6m/setup/N2_pyproject.diff): empty, parent and fix pyproject.toml identical |
| `harness/check.py` | - | 26ede77bdf82 | packages/sgl14956/harness/check.py (exact gate), docstring for this PR |
| `harness/gate.json` | - | a66df9aa2266 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b5_job.sh` (fecbbfb4bd8d) | 395b804fcfd5 | batch-4/5 guarded job (b5_job.sh = b4_job.sh), machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE; since 2026-11-25 it also requires R4_ROOT=$ISO_ROOT when it looks for its own processes (acc6: two copies with the same labels could kill each other's runs) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | packages/sgl14956/harness/mkvariant.sh, unchanged |
| `harness/ref_correct.json` | - | 68f08eabbdf9 | authors' reference correctness pass (from calibration) |
| `harness/run.sh` | - | 88fc500b5eaa | packages/sgl14956/harness/run.sh, unchanged |
| `loads/cal.env` | batch 4 `env/N2.env` (9acb27bda2da) | 39c2f5c740be | the screen's load; bookkeeping variables (ID, BENCH, JOB_ENV) replaced by the harness (CUDA_HOME and SGLANG_DISABLE_CUDNN_CHECK set by run.sh) |
| `loads/cor.env` | - | d50f74f5f3c2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_n4k.env` | - | 76b23c275f1a | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_r64.env` | - | 91bd3a9172c6 | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/fetch_cuda.sh` | - | e20f7c2f0ada | packages/sgl14956/setup/fetch_cuda.sh, unchanged |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/sgl14956/setup/get_model.sh, unchanged |
| `setup/setup_lock.sh` | - | d95af6e60816 | packages/sgl14956/setup/setup_lock.sh, unchanged |
| `setup/stage.sh` | - | f5fd13224a84 | packages/sgl14956/setup/stage.sh, unchanged |
| `site.env.example` | - | 2e033717e004 | packages/sgl14956/site.env.example, unchanged |
| `variants/apply.py` | - | 161cb4608fe0 | new: the broken / cheating variants |
