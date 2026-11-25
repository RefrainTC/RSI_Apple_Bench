# Provenance of package iso6994

Authors' copies: `/home/coder/work/p2-port/` on the 4090-server pods (NFS), hashed 2026-11-24; the paths below are
relative to that directory (`env/<file>` was `b3/setup/<file>`). `provenance/authors.sha256` lists the full hashes.
Files that differ do so only where a machine path became a variable from `env.sh` (`/home/coder/work/p2-port` ->
`$ISO_ROOT`, `/home/coder/work/rsi` -> `$ISO_ROOT`, `/tmp/p2` -> `$STAGE`, the uv/python binaries -> `$UV`/`$SYSPY`)
plus the small behaviour notes listed under the table; `provenance/paths.diff` shows every changed line. Comments that
still mention `/tmp/p2` or `/home/coder/...` describe the authors' layout and are not used.

| File | Authors' sha256 | Package sha256 | Status |
|---|---|---|---|
| `ENV.md` | - | 8a929fdeacdc | package documentation |
| `RUN.md` | - | 0337e4c0a63d | package documentation |
| `b3/bench/make_lora.py` | 4cd81439a92b | 4cd81439a92b | identical |
| `b3/bench/run_b3.sh` | 4a1ad7dbf617 | 0baf77b0f764 | machine paths parameterized (see paths.diff) |
| `b3/bench/sgl_b3.py` | 70a14c42de03 | 70a14c42de03 | identical |
| `b3/env/6994_cal.env` | db180e58153e | aec4d7f664ab | machine paths parameterized (see paths.diff) |
| `b3/env/6994_cor.env` | 59022b3b53c7 | 1a82ec76a119 | machine paths parameterized (see paths.diff) |
| `b3/env/6994_tr_2lora.env` | f1aaf68884c4 | 3a3af92bdc2a | machine paths parameterized (see paths.diff) |
| `b3/env/6994_tr_b128.env` | 5dfb9abe4f8a | 24365e21ff13 | machine paths parameterized (see paths.diff) |
| `b3/env/6994_tr_b16.env` | 1b129166743b | 851f50e2ce0d | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_cal.sh` | 19a0d3612501 | 0c369f7adfa0 | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_caldrv.sh` | ce83ced8c23d | 3bae0e21a60f | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_cpus.sh` | bc96242778c4 | c97093139ad3 | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_job.sh` | ca901b084675 | 27bbb7dfbcb9 | machine paths parameterized (see paths.diff) |
| `b3/scripts/b3_stage.sh` | 75cab67ffa2b | 4bdc65258a17 | machine paths parameterized (see paths.diff) |
| `b3/scripts/cal_analyze.py` | 36cb8bdc7747 | 36cb8bdc7747 | identical |
| `b3/scripts/get_model.sh` | - | dca6e0910056 | package addition: model download at the pinned revision (+ LoRA adapters for #6994), sha256 checks |
| `b3/scripts/hosttag.sh` | fdbe727e2408 | fdbe727e2408 | identical |
| `b3/scripts/pkg_stage.sh` | - | 76c99dfe13b9 | package addition: runs the authors' stage_tmp.sh py + b3_stage.sh in one step |
| `b3/scripts/setup_lock.sh` | - | 81f6a891b7b3 | package addition: builds the venvs from the hashed lock and checks them against the authors' records |
| `b3/scripts/setup_sgl.sh` | 72b8904c6aed | 1ba8d999b6aa | machine paths parameterized (see paths.diff) |
| `env.sh` | - | 4484cda2b9d7 | package addition: the one place for machine settings (ISO_ROOT, STAGE, UV, SYSPY; CPUS/PORT documented) |
| `env/6994_deps.lock` | - | 4f29178d9427 | generated 2026-11-24 from the authors' freeze: uv 0.11.14 pip compile --no-deps --generate-hashes with the authors' cutoff (and override) |
| `env/6994_fix_freeze.txt` | a369fe9e4a14 | a369fe9e4a14 | identical |
| `env/6994_fix_sgl.sha256` | 8c5184659ab0 | 8c5184659ab0 | identical |
| `env/6994_fix_src.sha256` | 0549ef030f54 | 0549ef030f54 | identical |
| `env/6994_freeze.diff` | 4d8fc754a8df | 4d8fc754a8df | identical |
| `env/6994_parent_freeze.txt` | 208a9f636987 | 208a9f636987 | identical |
| `env/6994_parent_sgl.sha256` | ead28fc1e940 | ead28fc1e940 | identical |
| `env/6994_parent_src.sha256` | 81e040452ccc | 81e040452ccc | identical |
| `env/6994_sgl_pkg_files.txt` | 3ab077447662 | 3ab077447662 | identical |
| `env/build.env` | - | fb4cb8df03fc | package addition: SHAs, merge date, wheel/tarball URLs and sha256, model revision (data) |
| `env/sgl_overrides.txt` | c14c7997f140 | c14c7997f140 | identical |
| `screen/with_env.sh` | fcf671aef6c8 | fcf671aef6c8 | identical |
| `scripts/stage_tmp.sh` | 5645fb591acc | f78afcfb16ff | machine paths parameterized (see paths.diff) |
