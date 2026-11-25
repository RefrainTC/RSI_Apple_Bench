# Provenance of package vllm27418

The task was screened in batch 5 (`raw/P2-port-4090/SCREEN_BATCH5.md`, candidate C77, "screened on main by c5y"; kit =
a copy of n1's `/home/coder/work/b5/` in `/home/coder/work/c5y/b5m/` on the main pod). This package was written on
2026-11-25 for the full task. Files derived from the batch-5 kit are listed with the original's sha256 (n1's copy; first 12
hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring numbers in the
card were produced by these same files (copied to `/home/coder/work/c5y/pkg/vllm27418` on the main pod), with the venvs
copied from the screen's build on main (`/tmp/c5y/b5m/venvs`; their freezes and vllm files equal `env/`); the
empty-directory test (`packages/VERIFY_batch5.md`) built everything from this package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 07f178f5d604 | documentation |
| `RUN.md` | - | 397a363f3a9c | documentation |
| `bench/offline_c5y.py` | batch 4 `bench/offline_b5.py` (49708d8ff8f7) | 96ff764ac514 | image mode (IMG) + the SEQ=1 switch added on main for the screen's load w2, unchanged (same images, text, seeds, warm-up, digests, metric); added: prompt token counts, settings.json, seqs.json, CORRECT (text pass), IMGPASS (image pass) and TF (teacher-forced re-read) modes; FIRST / text / ids modes dropped |
| `bench/run_bench.sh` | batch 4 `bench/run_b5.sh` (70db12a67111) | ca33e6e2440c | run_b5.sh + with_env.sh in one file, path from $ISO_ROOT (identical to packages/vllm37338/bench/run_bench.sh) |
| `env.sh` | - | a483ca272db9 | packages/vllm37338/env.sh, unchanged |
| `env/build.env` | - | ec037645a947 | new: commits, URLs, model revision (data) |
| `env/deps.lock` | - | f5e2ef7d1972 | generated 2026-11-25 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff merge date + 2 days); without vllm, flashinfer-python and triton-kernels (installed separately, env/build.env) |
| `env/files_differing.txt` | - | 80703e21a671 | authors' record (c5y/b5m/setup/C77_vllm_pkg_files.txt) |
| `env/fix_SOURCE.txt` | - | d70fe00c61c7 | authors' record of the wheel (URL, sha256, cudart): c5y/b5m/setup/C77_fix_SOURCE.txt |
| `env/fix_freeze.txt` | - | 362fe44171cc | authors' record: batch-5 screen build on main, c5y/b5m/setup/C77_fix_freeze.txt (equals n1's b5/setup apart from the wheel path, and n1's has one more line, wheel==0.45.1) |
| `env/fix_vllm.sha256` | - | b9ea83f4a3b5 | authors' record: c5y/b5m/setup/C77_fix_vllm.sha256 |
| `env/freeze.diff` | - | b9e8863dcf3d | authors' record (diff of the two freezes) |
| `env/parent_SOURCE.txt` | - | 39dc47e39a10 | authors' record of the wheel (URL, sha256, cudart): c5y/b5m/setup/C77_parent_SOURCE.txt |
| `env/parent_freeze.txt` | - | bc120eae46e4 | authors' record: batch-5 screen build on main, c5y/b5m/setup/C77_parent_freeze.txt (equals n1's b5/setup apart from the wheel path, and n1's has one more line, wheel==0.45.1) |
| `env/parent_vllm.sha256` | - | 834a9a4682ee | authors' record: c5y/b5m/setup/C77_parent_vllm.sha256 |
| `harness/check.py` | - | 5d21ddf258d3 | packages/vllm35442/harness/check.py: counts incl. image-token counts, settings, text pass exact, teacher-forced re-read (from packages/vllm37338/harness/check.py); timed digests reported, not gating |
| `harness/gate.json` | - | d4af089b3d40 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b5_job.sh` (27058cfaefb1) | 395b804fcfd5 | batch-4/5 guarded job (b5_job.sh = b4_job.sh), machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE; since 2026-11-25 it also requires R4_ROOT=$ISO_ROOT when it looks for its own processes (acc6: two copies with the same labels could kill each other's runs) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | packages/sgl27868/harness/mkvariant.sh, unchanged |
| `harness/ref_correct.json` | - | 98f0c1031d5f | authors' reference correctness pass (from calibration) |
| `harness/run.sh` | - | cac6544211ca | packages/vllm35442/harness/run.sh + image pass jobs and the teacher-forced re-read job (from packages/vllm37338); 2026-11-25 (acc6): the parent's cache is copied into a directory that now exists, a failed copy stops the job, and calibration runs the warm-ups before the correctness passes |
| `loads/cal.env` | batch 4 `env/C77.env` (2eb398295809) | 0c12352817af | the screen's C77w2.env (written on main, c5y/b5m/env: C77.env with 1344 x 1344 images, N=4, SEQ=1, OL=8, REPS=5); bookkeeping variables (ID, BENCH) replaced by the harness |
| `loads/cor.env` | - | 61ea7546113f | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/img.env` | - | 883bad704629 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tf.env` | - | e7f3cdd1823f | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_i448.env` | - | 78a5bfe2cb6f | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_w1.env` | - | de19b606f951 | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/vllm37338/setup/get_model.sh, unchanged |
| `setup/setup_lock.sh` | - | 52094a965032 | packages/vllm37338/setup/setup_lock.sh + the flashinfer sdist (no build isolation) and triton-kernels (git) steps; freeze comparison ignores the flashinfer line's URL |
| `setup/stage.sh` | - | 9fe070a7c2ac | packages/vllm37338/setup/stage.sh, unchanged |
| `site.env.example` | - | fc8fbcfad577 | packages/vllm37338/site.env.example, unchanged |
| `variants/apply.py` | - | 352f3ac6a261 | new: the broken / cheating variants |
