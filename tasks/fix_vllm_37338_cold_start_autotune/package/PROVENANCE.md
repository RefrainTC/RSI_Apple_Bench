# Provenance of package vllm37338

The task was screened and briefly calibrated in batch 4 (`raw/P2-port-4090/SCREEN_BATCH4.md`, candidate C19; second host
in `SCREEN_BATCH4_main.md`; scripts in `raw/P2-port-4090/screen_batch4/`, pod copy `/home/coder/work/b4/`). This package
was written on 2026-11-24/25 for the full task. Files derived from batch-4 scripts are listed with the original's sha256
(first 12 hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring
numbers in the card were produced by these same files (copied to `/home/coder/work/c5y/pkg/vllm37338` on the main pod),
with the venvs taken from batch 4's build (copied from main's `/tmp/b4`, itself a byte-identical copy of n1's; their
vllm files and freezes equal `env/`); the empty-directory test (`packages/VERIFY_batch4b.md`) built everything from this
package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | e4f1d5a5b209 | documentation |
| `RUN.md` | - | 577ec065c972 | documentation |
| `bench/offline_c5y.py` | batch 4 `bench/offline_b4.py` (0f7813e2b098) | 7155dc317e56 | FIRST=1 path unchanged (same prompts, seeds, digests); added: wall-clock stamps for the cold score, settings readback, seqs.json, TF mode; non-FIRST modes dropped |
| `bench/run_bench.sh` | batch 4 `bench/run_b4.sh` (76f7fc43e372) | ca33e6e2440c | run_b4.sh + with_env.sh in one file, path from $ISO_ROOT |
| `env.sh` | - | a483ca272db9 | packages/sgl16521/env.sh with this package's defaults (STAGE /tmp/c5y/st, PORT 36000 + 500 x GPU): caller's variables win over site.env |
| `env/build.env` | - | f79a163b9221 | new: commits, URLs, model revision (data) |
| `env/deps.lock` | - | 258c4314a666 | generated 2026-11-24 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff merge date + 2 days) |
| `env/files_differing.txt` | - | b8cc777851d2 | authors' record (batch 4 setup/C19_vllm_pkg_files.txt) |
| `env/fix_SOURCE.txt` | - | 0ce8a2843669 | authors' record of the wheel (URL, sha256, cudart), batch 4 setup/C19_fix_SOURCE.txt |
| `env/fix_freeze.txt` | - | aec69bf73a64 | authors' record, copied unchanged from batch 4 setup/C19_fix_freeze.txt |
| `env/fix_vllm.sha256` | - | 337f1d156207 | authors' record, copied unchanged from batch 4 setup/C19_fix_vllm.sha256 |
| `env/freeze.diff` | - | 582d8e903c06 | authors' record (batch 4 setup/C19_freeze.diff) |
| `env/parent_SOURCE.txt` | - | a91bf585de3c | authors' record of the wheel (URL, sha256, cudart), batch 4 setup/C19_parent_SOURCE.txt |
| `env/parent_freeze.txt` | - | 78dbeabb7751 | authors' record, copied unchanged from batch 4 setup/C19_parent_freeze.txt |
| `env/parent_vllm.sha256` | - | 088ca96fb11f | authors' record, copied unchanged from batch 4 setup/C19_parent_vllm.sha256 |
| `harness/audit.py` | - | 7e04e5248d2a | new: static audit of a candidate venv (only Python source may differ from the parent) |
| `harness/check.py` | - | 446de67bd02d | new: timing, output checks, gate, verdict |
| `harness/gate.json` | - | 300457ec19f0 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b4_job.sh` (aa2df9db2850) | 1975e9a638d4 | batch-4 guarded job, machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE (via packages/sgl16521/harness/job.sh, byte-identical) |
| `harness/mkvariant.sh` | - | c2b943a03de0 | new: broken-variant venvs (hard-linked copies + Python edits) |
| `harness/run.sh` | - | 2928c7c072aa | new: job lists of calibration / scoring / variants / transfers, STOP and lock handling; fresh caches, cold witness and venv snapshot per run |
| `loads/cal.env` | batch 4 `env/C19_cal.env` (44c23e2c7580) | 2aae217dd999 | same load with 3 steady reps instead of 2; FRESH_TRITON replaced by the harness's all-caches-fresh policy |
| `loads/smoke.env` | - | c0c6786cad6d | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tf.env` | - | ad5e5f3f5e57 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_long.env` | - | f98ba3244517 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_short.env` | - | bb1ff1ea9220 | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/sgl16521/setup/get_model.sh, unchanged: model download at the pinned revision, blob sha256 check, retries on HTTP 429 |
| `setup/setup_lock.sh` | - | c596e5282a34 | new (structure of packages/sgl16521/setup/setup_lock.sh): builds both venvs from the hashed lock and checks them against the authors' records |
| `setup/stage.sh` | - | 9fe070a7c2ac | packages/sgl16521/setup/stage.sh without the CUDA-tree step: copies interpreter, venvs, model to $STAGE and relocates the venvs |
| `site.env.example` | - | fc8fbcfad577 | new |
| `variants/apply.py` | - | 94dbe3d14aca | new: the broken / cheating variants |
