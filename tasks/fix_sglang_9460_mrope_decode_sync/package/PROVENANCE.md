# Provenance of package sgl9460

The task was screened and briefly calibrated in batch 4 (`raw/P2-port-4090/SCREEN_BATCH4.md`, candidate C47; second host
in `SCREEN_BATCH4_main.md`; scripts in `raw/P2-port-4090/screen_batch4/`, pod copy `/home/coder/work/b4/`). This package
was written on 2026-11-24/25 for the full task. Files derived from batch-4 scripts are listed with the original's sha256
(first 12 hex digits); the full hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring
numbers in the card were produced by these same files (copied to `/home/coder/work/c5y/pkg/sgl9460` on the main pod),
with the venvs taken from batch 4's build (copied from main's `/tmp/b4`, itself a byte-identical copy of n1's; their
sgl files and freezes equal `env/`); the empty-directory test (`packages/VERIFY_batch4b.md`) built everything from this
package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 33f21fbb9fc4 | documentation |
| `RUN.md` | - | 946f415a9430 | documentation |
| `bench/run_bench.sh` | batch 4 `bench/run_b4.sh` (76f7fc43e372) | 2ab0ed836f94 | run_b4.sh + with_env.sh in one file, path from $ISO_ROOT |
| `bench/sgl_c5y.py` | batch 4 `bench/sgl_b4.py` (669cc1b1576e) | 7c72622ee409 | timed IMG part unchanged (same images, text, seeds, digests, metric); added: seqs.json, settings.json, TEXT_ONLY, CORRECT (batch-1 correctness pass) and TF (HF teacher-forced re-read) modes; other modes dropped |
| `env.sh` | - | 815415d46e86 | packages/sgl16521/env.sh with this package's defaults (STAGE /tmp/c5y/st, PORT 36000 + 500 x GPU): caller's variables win over site.env |
| `env/build.env` | - | 8dec257e64c0 | new: commits, URLs, model revision (data) |
| `env/deps.lock` | - | 988fe8738375 | generated 2026-11-24 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff merge date + 2 days) |
| `env/files_differing.txt` | - | 36898dc2bdab | authors' record (batch 4 setup/C47_sgl_pkg_files.txt) |
| `env/fix_freeze.txt` | - | c110f3b70f94 | authors' record, copied unchanged from batch 4 setup/C47_fix_freeze.txt |
| `env/fix_sgl.sha256` | - | 9eec69636b6e | authors' record, copied unchanged from batch 4 setup/C47_fix_sgl.sha256 |
| `env/fix_src.sha256` | - | 1416386bb3b3 | authors' record, copied unchanged from batch 4 setup/C47_fix_src.sha256 |
| `env/freeze.diff` | - | af08f14a00d7 | authors' record (batch 4 setup/C47_freeze.diff) |
| `env/parent_freeze.txt` | - | 30ae36b6ff3e | authors' record, copied unchanged from batch 4 setup/C47_parent_freeze.txt |
| `env/parent_sgl.sha256` | - | ec29240663ab | authors' record, copied unchanged from batch 4 setup/C47_parent_sgl.sha256 |
| `env/parent_src.sha256` | - | 5044977b4259 | authors' record, copied unchanged from batch 4 setup/C47_parent_src.sha256 |
| `harness/check.py` | - | 0df7796f30e6 | new: timing, output checks, gate, verdict |
| `harness/gate.json` | - | b6b61e30aef5 | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b4_job.sh` (aa2df9db2850) | 1975e9a638d4 | batch-4 guarded job, machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE (via packages/sgl16521/harness/job.sh, byte-identical) |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | new: broken-variant venvs (hard-linked copies + Python edits) |
| `harness/ref_correct.json` | - | 08ecb8d1b1aa | authors' reference correctness pass: results/cal_g2/cor_1_parent/correct.json of the calibration |
| `harness/run.sh` | - | 7da0f70aa16e | new: job lists of calibration / scoring / variants / transfers, STOP and lock handling |
| `loads/cal.env` | batch 4 `env/C47_cal.env` (32a461c86797) | f7e71c116a3a | same load; batch-4 bookkeeping variables (ID, BENCH) replaced by the harness |
| `loads/cor.env` | - | 1e9c6f10968a | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tf.env` | - | b6deccf31376 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b32.env` | - | dcb603bcccd6 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_text.env` | - | fa9278412e4e | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/get_model.sh` | - | 02843fa38bdc | packages/sgl16521/setup/get_model.sh, unchanged: model download at the pinned revision, blob sha256 check, retries on HTTP 429 |
| `setup/setup_lock.sh` | - | d5a20b565bc0 | new (structure of packages/sgl16521/setup/setup_lock.sh): builds both venvs from the hashed lock and checks them against the authors' records |
| `setup/stage.sh` | - | 9fe070a7c2ac | packages/sgl16521/setup/stage.sh without the CUDA-tree step: copies interpreter, venvs, model to $STAGE and relocates the venvs |
| `site.env.example` | - | fc8fbcfad577 | new |
| `variants/apply.py` | - | 7b13c3fadc26 | new: the broken / cheating variants |
