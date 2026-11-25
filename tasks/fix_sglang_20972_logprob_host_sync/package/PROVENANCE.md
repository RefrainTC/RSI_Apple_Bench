# Provenance of package sgl20972

The task was screened and briefly calibrated in batch 4 (`raw/P2-port-4090/SCREEN_BATCH4.md`, candidate C53; scripts in
`raw/P2-port-4090/screen_batch4/`, pod copy `/home/coder/work/b4/`). This package was written on 2026-11-24/25 for the
full task. Files derived from batch-4 scripts are listed with the original's sha256 (first 12 hex digits); the full
hashes of every package file are in `MANIFEST.sha256`. The calibration and scoring numbers in the card were produced by
these same files (copied to `/home/coder/work/c5x/run20972` on the n1 pod), with the venvs taken from batch 4's build
(copied from n1's `/tmp/b4`; their sglang files and freezes equal `env/`); the empty-directory test
(`packages/VERIFY_batch4.md`) built everything from this package instead.

| File | Derived from | Package sha256 | Note |
|---|---|---|---|
| `ENV.md` | - | 7dd3da244cd0 | documentation |
| `RUN.md` | - | 00d83631814d | documentation |
| `bench/run_bench.sh` | batch 4 `bench/run_b4.sh` (76f7fc43e372) | 768b59f40fca | run_b4.sh + with_env.sh in one file, path from $ISO_ROOT |
| `bench/sgl_c5x.py` | batch 4 `bench/sgl_b4.py` (669cc1b1576e) | 7725a74d168a | timed part unchanged (same prompts, seeds, digests, metric); added: seqs.json and settings.json, CORRECT and TF modes; CLP/IMG modes dropped |
| `env.sh` | - | a3ba5363c6cd | new: the one place for machine settings; caller's variables win over site.env |
| `env/build.env` | - | f61a1b5f01b2 | new: commits, URLs, model revision (data) |
| `env/cuda_tree.sha256` | - | 49c5d255f5ee | generated from the authors' staged CUDA tree (/tmp/c5x/st/cuda, a copy of batch 4's /tmp/b4/cuda) |
| `env/deps.lock` | - | da362f11e239 | generated 2026-11-24 from the authors' parent freeze (uv pip compile --no-deps --generate-hashes, cutoff merge date + 2 days) |
| `env/files_differing.txt` | - | 6478cc27403a | authors' record (batch 4 setup/C53_sgl_pkg_files.txt) |
| `env/fix_freeze.txt` | - | 3a48ae84fb2f | authors' record, copied unchanged from batch 4 setup/C53_fix_freeze.txt |
| `env/fix_sgl.sha256` | - | 9abbee085ed4 | authors' record, copied unchanged from batch 4 setup/C53_fix_sgl.sha256 |
| `env/fix_src.sha256` | - | 9e251ef7b6ae | authors' record, copied unchanged from batch 4 setup/C53_fix_src.sha256 |
| `env/freeze.diff` | - | 5a97beff640c | authors' record (batch 4 setup/C53_freeze.diff) |
| `env/parent_freeze.txt` | - | d83e2b541cb4 | authors' record, copied unchanged from batch 4 setup/C53_parent_freeze.txt |
| `env/parent_sgl.sha256` | - | fec9f1ecdb40 | authors' record, copied unchanged from batch 4 setup/C53_parent_sgl.sha256 |
| `env/parent_src.sha256` | - | cc371717f160 | authors' record, copied unchanged from batch 4 setup/C53_parent_src.sha256 |
| `harness/check.py` | - | 11f5e73220e7 | new: timing (as cal_b4.py / cal_analyze.py), output checks, gate, verdict |
| `harness/gate.json` | - | 1c8e05928f3e | new: REF, band and thresholds set from the calibration (see the card) |
| `harness/job.sh` | batch 4 `scripts/b4_job.sh` (aa2df9db2850) | 1975e9a638d4 | batch-4 guarded job, machine paths -> env.sh variables (CPUS, STAGE, GPU), scratch under $STAGE, no NAS labels |
| `harness/mkvariant.sh` | - | 50e6cfa15f59 | new: broken-variant venvs (hard-linked copies + Python edits) |
| `harness/run.sh` | - | f95b6ae6e5fa | new: job lists of the calibration / scoring (structure of the b3 packages' b3_cal.sh), STOP and lock handling |
| `loads/cal.env` | batch 4 `env/C53.env` (cd34665b1ef5) | 8b79eaee5fde | same load; batch-4 bookkeeping variables (ID, BENCH, JOB_ENV) replaced by the harness |
| `loads/cor.env` | - | 652ea1566f29 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/smoke.env` | - | 3c64d3f4cbf2 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tf.env` | - | f077d00ff780 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b1.env` | - | f410c1bc5819 | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_b32.env` | - | 3fb996a45b3a | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_top0.env` | - | 4993bba6e94b | load file (official load in `cal.env`; others source it and change one knob) |
| `loads/tr_top20.env` | - | 0949fab11e76 | load file (official load in `cal.env`; others source it and change one knob) |
| `setup/fetch_cuda.sh` | batch 4 `scripts/fetch_cuda.sh` (1fb53eef7994) | fb035b2ae8fd | same archives and sha256 pins; adds the venv headers step that batch 4 did by hand, and a tree hash check |
| `setup/get_model.sh` | - | 02843fa38bdc | new: model download at the pinned revision, blob sha256 check, retries on HTTP 429 |
| `setup/setup_lock.sh` | - | d5a20b565bc0 | new: builds both venvs from the hashed lock and checks them against the authors' records |
| `setup/stage.sh` | - | 0e1274233a76 | new: copies interpreter, venvs, model, CUDA tree to $STAGE and relocates the venvs |
| `site.env.example` | - | 769498501884 | new |
| `variants/apply.py` | - | 670de6bc9788 | new: the broken / cheating variants |
