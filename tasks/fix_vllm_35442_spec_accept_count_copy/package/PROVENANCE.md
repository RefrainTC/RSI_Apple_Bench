# Provenance of package vllm35442

The task was screened in batch 5 (`raw/P2-port-4090/SCREEN_BATCH5.md`, candidate C15, "screened on main by c5y"; kit =
a copy of n1's `/home/coder/work/b5/` in `/home/coder/work/c5y/b5m/` on the main pod). This package was written on
2026-11-25 for the full task. Files derived from the batch-5 kit are listed with the original's sha256 (n1's copy; first 12
hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring numbers in the
card were produced by these same files (copied to `/home/coder/work/c5y/pkg/vllm35442` on the main pod), with the venvs
copied from the screen's build on main (`/tmp/c5y/b5m/venvs`; their freezes and vllm files equal `env/`); the
empty-directory test (`packages/VERIFY_batch5.md`) built everything from this package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | ed83a4eb30d1 | documentation |
| `RUN.md` | - | 32f1f6ef8550 | documentation |
| `bench/offline_c5y.py` | batch 4 `bench/offline_b5.py` (49708d8ff8f7) | ac41924094d8 | default ids mode unchanged (same prompts, seeds, warm-up, digests, metric); added: settings.json, ids.json, CORRECT (correctness pass) mode; FIRST / text / image modes dropped |
| `bench/run_bench.sh` | batch 4 `bench/run_b5.sh` (70db12a67111) | ca33e6e2440c | run_b5.sh + with_env.sh in one file, path from $ISO_ROOT (identical to packages/vllm37338/bench/run_bench.sh) |
| `env.sh` | - | a483ca272db9 | packages/vllm37338/env.sh, unchanged |
| `env/build.env` | - | 19ee0b35e9da | new: commits, URLs, model revision (data) |
| `env/deps.lock` | - | 499c735557e5 | generated 2026-11-25 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff merge date + 2 days) |
| `env/files_differing.txt` | - | c40f51aed99a | authors' record (c5y/b5m/setup/C15_vllm_pkg_files.txt) |
| `env/fix_SOURCE.txt` | - | 02c3a8dd7c88 | authors' record of the wheel (URL, sha256, cudart): c5y/b5m/setup/C15_fix_SOURCE.txt |
| `env/fix_freeze.txt` | - | e8165b099066 | authors' record: batch-5 screen build on main, c5y/b5m/setup/C15_fix_freeze.txt (equals n1's b5/setup apart from the wheel path) |
| `env/fix_vllm.sha256` | - | c3dd035ed34b | authors' record: c5y/b5m/setup/C15_fix_vllm.sha256 |
| `env/freeze.diff` | - | 9e8fc4a32c68 | authors' record (diff of the two freezes) |
| `env/parent_SOURCE.txt` | - | 85698814cd16 | authors' record of the wheel (URL, sha256, cudart): c5y/b5m/setup/C15_parent_SOURCE.txt |
| `env/parent_freeze.txt` | - | 4083f6449ab5 | authors' record: batch-5 screen build on main, c5y/b5m/setup/C15_parent_freeze.txt (equals n1's b5/setup apart from the wheel path) |
| `env/parent_vllm.sha256` | - | aef36d5cb194 | authors' record: c5y/b5m/setup/C15_parent_vllm.sha256 |
| `harness/check.py` | - | 0f749982a485 | packages/sgl27868/harness/check.py (exact gate), docstring for this PR; logprobs may be missing (None) in the correctness pass |
| `harness/gate.json` | - | c9844dc5b500 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b5_job.sh` (27058cfaefb1) | 395b804fcfd5 | batch-4/5 guarded job (b5_job.sh = b4_job.sh), machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE; since 2026-11-25 it also requires R4_ROOT=$ISO_ROOT when it looks for its own processes (acc6: two copies with the same labels could kill each other's runs) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | packages/sgl27868/harness/mkvariant.sh, unchanged |
| `harness/ref_correct.json` | - | bdce05a91d6e | authors' reference correctness pass (from calibration) |
| `harness/run.sh` | - | d9c6842348d5 | packages/sgl27868/harness/run.sh with vLLM's cache and port variables: job lists of calibration / scoring / variants / transfers, STOP and lock handling; 2026-11-25 (acc6): the parent's cache is copied into a directory that now exists, a failed copy stops the job, and calibration runs the warm-ups before the correctness passes |
| `loads/cal.env` | batch 4 `env/C48.env` (9b7b87c60e16) | c527db3e457e | not derived: the screen's C15_n1.env (written on main, c5y/b5m/env) = C15 batch-1 transfer load; bookkeeping variables (ID, BENCH) replaced by the harness |
| `loads/cor.env` | - | de95cffb4355 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b16.env` | - | 34569083b4e2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b4.env` | - | efb3e19e79e8 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_idstats.env` | - | fb128cace522 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_text.env` | - | bb4916c29207 | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/vllm37338/setup/get_model.sh, unchanged |
| `setup/setup_lock.sh` | - | c596e5282a34 | packages/vllm37338/setup/setup_lock.sh, unchanged |
| `setup/stage.sh` | - | 9fe070a7c2ac | packages/vllm37338/setup/stage.sh, unchanged |
| `site.env.example` | - | fc8fbcfad577 | packages/vllm37338/site.env.example, unchanged |
| `variants/apply.py` | - | a8e3d5247057 | new: the broken / cheating variants |
