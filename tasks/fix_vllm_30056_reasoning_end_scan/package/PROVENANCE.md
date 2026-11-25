# Provenance of package vllm30056

The task was screened in batch 6 (`raw/P2-port-4090/SCREEN_BATCH6.md`, candidate N1; screened on main by c5y with the batch-6 kit
`/home/coder/work/c5y/b6m/`, a copy of `c5y/b5m/`; see `SCREEN_BATCH6.md`). This package was written on
2026-11-25 for the full task. Files derived from the batch-6 kit are listed with the original's sha256 (main's c5y/b6m copy; first 12
hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring numbers in the
card were produced by these same files (copied to `/home/coder/work/c5y/pkg/vllm30056` on the main pod), with the venvs
copied from the screen's build on main (`/tmp/c5y/b6m/venvs`; their freezes and vllm files equal `env/`); the
empty-directory test (`packages/VERIFY_batch6.md`) built everything from this package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 34304bce446b | documentation |
| `RUN.md` | - | e3678756dd22 | documentation |
| `bench/offline_c5y.py` | batch 4 `bench/offline_b5.py` (51cbdf91d07d) | 6da292796377 | batch-6 kit's offline_b5.py: default ids mode with SP_KW unchanged (same prompts, seeds, warm-up, digests, metric); added: settings.json (incl. structured_outputs_config), ids.json, CORRECT (chat-format correctness pass with thinking and a JSON schema); FIRST / text / image modes dropped |
| `bench/run_bench.sh` | batch 4 `bench/run_b5.sh` (76e552816138) | ca33e6e2440c | run_b5.sh + with_env.sh in one file, path from $ISO_ROOT (identical to packages/vllm37338/bench/run_bench.sh) |
| `env.sh` | - | a483ca272db9 | packages/vllm35442/env.sh, unchanged |
| `env/build.env` | - | 2fb8b59945dc | new: commits, URLs, model revision (data) |
| `env/deps.lock` | - | 9817017d9033 | generated 2026-11-25 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff 2025-12-11) |
| `env/files_differing.txt` | - | 218c42e18a7b | authors' record (c5y/b6m/setup/N1_vllm_pkg_files.txt) |
| `env/fix_SOURCE.txt` | - | 2517daf193b1 | authors' record of the wheel (URL, sha256, cudart): c5y/b6m/setup/N1_fix_SOURCE.txt |
| `env/fix_freeze.txt` | - | b227344d4d06 | authors' record: batch-6 screen build on main, c5y/b6m/setup/N1_fix_freeze.txt |
| `env/fix_vllm.sha256` | - | a1fbdb86bc68 | authors' record: c5y/b6m/setup/N1_fix_vllm.sha256 |
| `env/freeze.diff` | - | 2f6b073e7d5f | authors' record (diff of the two freezes) |
| `env/parent_SOURCE.txt` | - | d4ff2da509b9 | authors' record of the wheel (URL, sha256, cudart): c5y/b6m/setup/N1_parent_SOURCE.txt |
| `env/parent_freeze.txt` | - | 05198a7d8ea4 | authors' record: batch-6 screen build on main, c5y/b6m/setup/N1_parent_freeze.txt |
| `env/parent_vllm.sha256` | - | aba50265f744 | authors' record: c5y/b6m/setup/N1_parent_vllm.sha256 |
| `harness/check.py` | - | 6e40bbf3f8e7 | packages/vllm35442/harness/check.py (exact gate), docstring for this PR; reports how many correctness prompts ended their reasoning |
| `harness/gate.json` | - | 6a362fb6cd21 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b5_job.sh` (fecbbfb4bd8d) | 395b804fcfd5 | batch-4/5 guarded job (b5_job.sh = b4_job.sh), machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE; since 2026-11-25 it also requires R4_ROOT=$ISO_ROOT when it looks for its own processes (acc6: two copies with the same labels could kill each other's runs) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | packages/vllm35442/harness/mkvariant.sh, unchanged |
| `harness/ref_correct.json` | - | 2675a0a87d16 | authors' reference correctness pass (from calibration) |
| `harness/run.sh` | - | d9c6842348d5 | packages/vllm35442/harness/run.sh, unchanged (candidate runs start from a copy of the parent's compile cache); 2026-11-25 (acc6): the parent's cache is copied into a directory that now exists, a failed copy stops the job, and calibration runs the warm-ups before the correctness passes |
| `loads/cal.env` | batch 4 `env/N1.env` (4cde56a043fb) | ec809dbcbac0 | the screen's load; bookkeeping variables (ID, BENCH) replaced by the harness |
| `loads/cor.env` | - | 53810dbc2924 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b8.env` | - | cfb04e633d01 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_nogrammar.env` | - | 91b07b0e2f5a | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/vllm35442/setup/get_model.sh, unchanged |
| `setup/setup_lock.sh` | - | c596e5282a34 | packages/vllm35442/setup/setup_lock.sh, unchanged |
| `setup/stage.sh` | - | 9fe070a7c2ac | packages/vllm35442/setup/stage.sh, unchanged |
| `site.env.example` | - | fc8fbcfad577 | packages/vllm35442/site.env.example, unchanged |
| `variants/apply.py` | - | 2e2551386eb8 | new: the broken / cheating variants |
