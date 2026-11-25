# Kit validation on the RTX 5090 (2026-11-25)

Package sgl16521 (c5x family) was built fresh from a copy of the package in `/tmp/kit/sgl16521`
(`STAGE=/tmp/kit/st`, GPU 0, CPUs 0-7,16-23, `TZ=UTC`, uv 0.11.14). Setup checks all passed: `SETUP_LOCK_OK`,
`GET_MODEL_OK`, `FETCH_CUDA_OK`, `STAGE_OK`. Every GPU job ran under the machine lock
(`raw/gpu.lock`), interleaved with another agent's job chain, with a whole-device VRAM watchdog at 30 GiB. Peak VRAM
was 26.8 GB in every job. Candidates are Python-only edits.

## Did the 5090 REF come out as expected?

Yes. The fix scored 3.4956 in the calibration and 3.5203 and 3.5148 in two scoring jobs; every check passed in all
three. The calibration's A/A max deviation was 0.68 %. `set_ref.py cal_g0 s1_g0 s2_g0` wrote REF **3.5148**, SD 0.37 %,
and band **0.74 %** (2 SD, above the 0.5 % floor), plus `gpu_name` "NVIDIA GeForce RTX 5090" and the date. The shipped
4090 values (REF 3.827, band 1.26 %) are kept under `reference_rtx4090`. The unchanged `harness/check.py` reads the new
file. An earlier 5090 measurement by another agent gave 3.52; this one gives 3.51.

## Did test.sh score the three candidates as expected?

| Candidate | Input to test.sh | Speedup (pairs) | Verdict | Reward | As expected? |
|---|---|---|---|---|---|
| (a) untouched parent tree | the agent view's engine directory | 0.9999 (1.0023, 0.9999, 0.9996) | `no_gain` | 0.0 | yes (~1.0, no_gain) |
| (b) parent + the human fix's changed file | unified diff (`git apply -p4`) | 3.5126 (3.5224, 3.5126, 3.5115) | `ties_ref` | 0.999 | yes (ties_ref; REF 3.5148 ± 0.74 %) |
| (c) shipped broken variant `drophist` | directory with the variant's `sglang/` | 3.9559 (3.9372, 3.9559, 3.9572) | `fail:tf` | 0.0 | yes (fails the gate despite being fastest) |

For (c), the teacher-forced check found a maximum gap of 5.19 nats (the limit is 0.5) and up to 161 tokens below the
top-1 per run. The parent's own figures were 0.125 nats and 23. All kit checks passed for (a), (b) and (c):
- the parent engine before the lock, under the lock and after the job;
- the package files;
- the seal before and after;
- the trusted trees after each of the 11 runs (80,465 files);
- the 11 frozen result directories;
- no leftover candidate process.

For (c) the only failed check is `harness_gate`.

The kit's overhead per candidate was about 35 s: sealing the parent once took 12 s; seal verification takes 2-3 s;
building the candidate (an 11 GB real copy) takes ~20 s. The scoring job itself took 8.5-9 min, and each `test.sh`
took 9-9.5 min end to end.

## Does the agent view work on its own and stay clean?

- `make_agent_view.sh /tmp/kit/view16521` took 55 s and produced 14 GB.
- `dev_bench.sh` ran from the view alone and measured 0.3246 s (reps 0.3283, 0.3241, 0.3250, 0.3242). The
  harness's parent arm read 0.3241-0.3255 s in the same hour, so it times the same thing. It printed nothing about
  the fix or REF.
- The kit's own scan found 0 hard hits. It covered 11 files the kit wrote and 74,423 venv/interpreter files outside
  the engine.
- An independent grep (`/tmp/kit/logs/grep_proof.log`) found none of the following in the files the kit wrote: the
  fix commit (full or 7 characters), `radix_cache.py`, `mem_cache/radix_cache`, 3.5148, 3.827, `gate.json`,
  `RUN.md`, `PROVENANCE`, `files_differing`, `variants`, `16521_fix`, `_fix`, "human fix", "maintainer", `REF`.
- Across the whole view (binary files included), the fix commit, `16521_fix` and the package path occur 0 times.
- The view's engine hashes as the parent's in all 1857 files; its `radix_cache.py` is the parent's (285974be…), not
  the fix's (9f3112d6…).
- The view shares 0 inodes with the operator's parent venv and STAGE.
- Remaining soft hit, for review: `bench/sgl_c5x.py` keeps the engine argument `disable_radix_cache=True` in its
  teacher-forced branch. This is code, not a comment, and the public task text already describes prefix reuse.
- PR number: 0 occurrences in the files the kit wrote. The view directory name chosen here contains it, so sysconfig
  paths do too. Choose a neutral directory name.

## Does the parent stay untouched by edits in the view or a candidate?

Four files were appended to in place: in the view, `sglang/.../radix_cache.py` and `torch/__init__.py`; in a
candidate venv, `sglang/.../scheduler.py` and `torch/__init__.py`. Each had link count 1. Afterwards the parent's 1857
engine files still hashed as recorded, and the parent's `torch/__init__.py` did not contain the line.

`make_candidate.sh` refused each of these:
- an added `.so`: "only Python source (*.py) may differ";
- a syntax error: "candidate does not byte-compile";
- the reserved name `parent`;
- the parent venv itself given as the agent tree.

It left no venv or marker behind in any of these cases.

The monitor's synthetic test (`/tmp/kit/bin/monitor_selftest.sh`) caught three patterns:
- a trusted file changed in one candidate run and restored in a later one (flagged after every run in between);
- a parent run's `latency.json` rewritten after it finished;
- a process left behind by a candidate run.

## Does test.sh refuse when it should?

| Case | test.sh outcome | Reward file |
|---|---|---|
| `gate.json` without `gpu_name` (shipped file, before set_ref) | exit 3, "REFUSED: harness/gate.json has no gpu_name …" | reward 0, `refused:uncalibrated` |
| `gate.json` `gpu_name` set to "NVIDIA GeForce RTX 4090" | exit 3, "REFUSED: … calibrated on 'NVIDIA GeForce RTX 4090' but GPU 0 is 'NVIDIA GeForce RTX 5090' …" | reward 0, `refused:gpu_mismatch` |
| parent engine file `sglang/srt/server_args.py` edited in place | exit 3, "REFUSED: PARENT ENGINE FILES DIFFER … server_args.py: FAILED" | reward 0, `refused:tampered` (`parent_engine_pre`) |
| `gate.json` REF halved (same GPU name) | exit 3, "REFUSED: PACKAGE FILES DIFFER: harness/gate.json differs from the set_ref.py record" | reward 0, `refused:tampered` (`package_files_pre`) |
| parent non-engine file `torch/__init__.py` edited in place | exit 3 after taking the lock, "REFUSED: TRUSTED TREES DIFFER FROM THE SEAL: … torch/__init__.py" (the engine hash check cannot see this file; the seal does) | reward 0, `refused:tampered` (`seal_pre`) |

Every edit was restored afterwards, and the parent engine, the package files and the seal verified clean again.

## Does the kit work on the c5y family?

Tested on sgl20503 (c5y-sglang), with no GPU:
- The package was built fresh (`SETUP_LOCK_OK`, `STAGE_OK`).
- `make_agent_view.sh` took 70 s and produced 12 GB with 0 hard hits. It extracted the timed-run environment from
  the package's `run.sh` (`XDG_CACHE_HOME`, `SGLANG_DISABLE_CUDNN_CHECK=1`).
- `make_candidate.sh` worked from the fix diff: the candidate's `scheduler.py` equals the fix record (6f2c1d97…).
- `make_candidate.sh` also worked from the view's untouched tree.
- An in-place edit in that candidate left the parent intact.

A static check over all 11 packages (`/tmp/kit/logs/static_adapter.log`) passed everywhere: family and engine
detection, extraction of the timed-run environment from `run.sh` (vllm37338 detected as cold caches), comment stripping
with the AST check, and the loads' path references.

## Where are the logs and state?

Logs are in `/tmp/kit/logs/`:
- `build16521.log`, `16521_{cal_g0,s1_g0,s2_g0}.log`, `devbench.log`, `set_ref.log`;
- `test_kit{a,b,c}.log`, `tamper_checks.log`, `nongpu_checks.log`, `monitor_selftest.log`;
- `grep_proof.log`, `c5y_checks.log`, `static_adapter.log`, `gpujobs.log`.

Reward files are in `/tmp/kit/rewards/`. Job results are in `/tmp/kit/sgl16521/results/`. The monitor reports, the
seal, the gate record and the scan reports are in `/tmp/kit/sgl16521/kit_state/`.

## Follow-up: c5y-sglang family (sgl20503), RTX 5090, 2026-11-25

Built fresh from the package and checked against its manifest. Kit copied from this folder. The whole chain ran under one hold of the GPU lock.

| Step | Result |
|---|---|
| Agent view | 12 GB in 64 s; leak scan **0 hard hits**. The only soft hits were the digits "20503" inside third-party files (scipy, flashinfer checksums). |
| Seal of the parent (4 trusted trees, 80,507 files) | made in 13 s; **unchanged after a full calibration** (`seal ok`). Parent runs do not write into the parent venv, interpreter, model or CUDA tree, so the monitor raises no false tamper alarms on this family. |
| Calibration `cal_g0` | 1.1300 (`above_ref` against the shipped 4090 REF 1.08) |
| Fix scoring `s1_g0` | 1.1447 |
| `set_ref.py cal_g0 s1_g0` | RTX 5090 REF **1.1374**, SD 0.91 %, band ±0.0183 (1.6 %) |
| `dev_bench.sh` from the view | median rep 11.97 s; peak VRAM 26.9 GB |
| `test.sh` (a) untouched tree | **1.0001, `no_gain`, reward 0.0005** (pairs 1.0086 / 0.9972 / 1.0001); monitor: 0 tree changes, 0 violations |
| `test.sh` (b) human fix as a unified diff | **1.1206, `ties_ref`, reward 0.877** (pairs 1.1019 / 1.1242 / 1.1206); monitor: 0 tree changes, 0 violations |

The human fix gains more on the RTX 5090 (1.137) than on the RTX 4090 (1.08). This is another case where each GPU type needs its own REF.
