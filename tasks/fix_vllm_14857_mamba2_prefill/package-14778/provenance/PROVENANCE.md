# Provenance of package iso14778

Authors' copies: `/home/coder/work/p2-port/` on the 4090-server pods (NFS), hashed 2026-11-24; the paths below are
relative to that directory (`env/<file>` was `b3/setup/<file>`). `provenance/authors.sha256` lists the full hashes.
Files that differ do so only where a machine path became a variable from `env.sh` (`/home/coder/work/p2-port` ->
`$ISO_ROOT`, `/home/coder/work/rsi` -> `$ISO_ROOT`, `/tmp/p2` -> `$STAGE`, the uv/python binaries -> `$UV`/`$SYSPY`)
plus the small behaviour notes listed under the table; `provenance/paths.diff` shows every changed line. Comments that
still mention `/tmp/p2` or `/home/coder/...` describe the authors' layout and are not used.

| File | Authors' sha256 | Package sha256 | Status |
|---|---|---|---|
| `ENV.md` | - | 04a7bd79ce91 | package documentation |
| `RUN.md` | - | 297324a6d392 | package documentation |
| `b3/bench/offline_b3.py` | a0af124c8e5f | a0af124c8e5f | identical |
| `b3/bench/run_b3.sh` | 4a1ad7dbf617 | 0baf77b0f764 | machine paths parameterized (see paths.diff) |
| `b3/env/14778_cal.env` | fba6b843ddf8 | fba6b843ddf8 | identical |
| `b3/env/14778_cor.env` | 204b8ea759ad | 204b8ea759ad | identical |
| `b3/env/14778_tr_in256.env` | 1d830f21ac70 | 1d830f21ac70 | identical |
| `b3/env/14778_tr_ol64.env` | f7ed05e912c5 | f7ed05e912c5 | identical |
| `b3/env/14778_tr_seqs16.env` | 378caa6e44f1 | 378caa6e44f1 | identical |
| `b3/scripts/b3_cal.sh` | 19a0d3612501 | 0c369f7adfa0 | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_caldrv.sh` | ce83ced8c23d | 3bae0e21a60f | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_cpus.sh` | bc96242778c4 | c97093139ad3 | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_job.sh` | ca901b084675 | 27bbb7dfbcb9 | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_stage.sh` | 75cab67ffa2b | 4bdc65258a17 | machine paths parameterized (see paths.diff) |
| `b3/scripts/cal_analyze.py` | 36cb8bdc7747 | 36cb8bdc7747 | identical |
| `b3/scripts/get_model.sh` | - | dca6e0910056 | package addition: model download at the pinned revision (+ LoRA adapters for #6994), sha256 checks |
| `b3/scripts/hosttag.sh` | fdbe727e2408 | fdbe727e2408 | identical |
| `b3/scripts/pkg_stage.sh` | - | 76c99dfe13b9 | package addition: runs the authors' stage_tmp.sh py + b3_stage.sh in one step |
| `b3/scripts/setup_b3.sh` | 99c2c8ae37a9 | d593dfb978a9 | machine paths parameterized (see paths.diff) |
| `b3/scripts/setup_lock.sh` | - | 81f6a891b7b3 | package addition: builds the venvs from the hashed lock and checks them against the authors' records |
| `env.sh` | - | 48a2194c775a | package addition: the one place for machine settings (ISO_ROOT, STAGE, UV, SYSPY; CPUS/PORT documented) |
| `env/14778_deps.lock` | - | d6967fdc481d | generated 2026-11-24 from the authors' freeze: uv 0.11.14 pip compile --no-deps --generate-hashes with the authors' cutoff (and override) |
| `env/14778_fix_freeze.txt` | 41d63e7b78a0 | 41d63e7b78a0 | identical |
| `env/14778_fix_vllm.sha256` | f4bff7c1bd7e | f4bff7c1bd7e | identical |
| `env/14778_freeze.diff` | 9df40a3d1b36 | 9df40a3d1b36 | identical |
| `env/14778_parent_freeze.txt` | 8cd055471df0 | 8cd055471df0 | identical |
| `env/14778_parent_vllm.sha256` | f8e6c095e930 | f8e6c095e930 | identical |
| `env/14778_vllm_pkg_files.txt` | 1ed0e5d17ceb | 1ed0e5d17ceb | identical |
| `env/build.env` | - | 9c65a5b55b3c | package addition: SHAs, merge date, wheel/tarball URLs and sha256, model revision (data) |
| `env/uv_overrides_14778.txt` | 7f93885c01f7 | 7f93885c01f7 | identical |
| `screen/fetch.py` | f0c30dae1a74 | f0c30dae1a74 | identical |
| `screen/with_env.sh` | fcf671aef6c8 | fcf671aef6c8 | identical |
| `scripts/stage_tmp.sh` | 5645fb591acc | f78afcfb16ff | machine paths parameterized (see paths.diff) |
