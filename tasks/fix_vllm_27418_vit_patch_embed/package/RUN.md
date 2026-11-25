# Run vLLM #27418 (Qwen2-VL vision tower: patch embedding Conv3d -> Linear) from this package

- PR: https://github.com/vllm-project/vllm/pull/27418, merged 2025-10-24.
- Human fix `42efe609ba75eb1b0bc06ae635778b2bc0aa4e7a`; parent `88d3141ec63008d8dff63311359355ebbe98ad64`.
- Card: `CARD_vllm27418.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps go from an empty Linux machine
to one calibration and one scoring job of the human fix (what the card reports), plus the broken variants that the
checks must reject.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s, driver 550 = CUDA 12.4; torch 2.9.0+cu128 runs
  on it). The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `git` (triton-kernels is a git dependency), `flock`, `taskset`, `nvidia-smi`, a host `python3`
  >= 3.8 (stdlib only). No CUDA toolkit, no compiler (the flashinfer sdist builds as pure Python).
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`).
- Internet access to pypi.org, files.pythonhosted.org, wheels.vllm.ai, github.com, huggingface.co.
- Disk: ~30 GB under `$ISO_ROOT` (two venvs ~8 GB each, uv cache, model 4.4 GB), ~25 GB under `$STAGE`.

## 1. Settings

```
cp -r vllm27418 /data/vllm27418 && cd /data/vllm27418   # any location; this is ISO_ROOT
cp site.env.example site.env                           # then edit: STAGE, GPU, CPUS, PORT (UV if uv is not on PATH)
```
Variables you set in the shell win over `site.env` (e.g. `GPU=1 CPUS=8-15,40-47 PORT=36500 bash harness/run.sh ...`).
**CPU sets are host-specific**: pick the cores near the GPU (`nvidia-smi topo -m`, column "CPU Affinity"). On the
authors' hosts: main pod GPU0 `0-7,32-39`, GPU1 `8-15,40-47`, GPU2 `16-23,48-55`, GPU3 `24-31,56-63`; n1 pod
GPU1 `24-31,56-63`. On these pods `nvidia-smi topo -m` shows affinity `0-63` for every GPU, so it cannot tell you the
set: the table above is the authors' own mapping (one CCD per GPU), and on another host you have to find the cores near
each GPU from its hardware documentation or `lscpu`. Omitting `CPUS` runs unpinned (noisier).
Keep `$STAGE` short (< ~20 characters; IPC socket paths are limited to 107 characters) and on local disk, not NFS.

## 2. Build the two venvs (CPU and network; ~5-15 min)

```
bash setup/setup_lock.sh
```
Must end with `SETUP_LOCK_OK 27418`, after printing for both roles that the freeze equals the authors', that all vllm
files hash as in the authors' build, and the import check. Besides the hashed lock it installs flashinfer-python 0.4.1
from its hash-pinned sdist without build isolation and triton-kernels from git (`ENV.md`). The vision files of six
models (`env/files_differing.txt`) differ between parent and fix.

## 3. Model (~3-10 min)

```
bash setup/get_model.sh      # Qwen/Qwen2-VL-2B-Instruct @ 1860c46f...; ends with GET_MODEL_OK
```

## 4. Stage to local disk (~2-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 27418
```

## 5. Calibration, then a scoring job of the human fix (one GPU; ~30 + ~15 min)

```
bash harness/run.sh cal_g0 cal          # 28 runs + the teacher-forced re-read
bash harness/run.sh s1_g0 score         # 11 runs + the teacher-forced re-read; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`).
Compile caches are kept per role, host and GPU under `$STAGE/cache/`; every candidate run starts from a fresh copy of
the parent's cache on that GPU, so both arms run the same tuned kernels (see the card, "The gate"). Run the parent of a
GPU first (every mode does). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
A second GPU can run another job at the same time: `GPU=1 CPUS=<cores near GPU1> PORT=36500 bash harness/run.sh s2_g1 score`.
Transfer loads (448 x 448 images one per call; the screen's first load, 32 images in one call): `bash harness/run.sh tr_g0 transfer`.
Their outputs are not re-read (the re-read rebuilds 1344 x 1344 images only); their digests are reported.
To re-derive a verdict with other thresholds: edit `harness/gate.json`, then `python3 -I harness/check.py $ISO_ROOT <label>`.


**Compile caches.** Every candidate run starts from a copy of the parent's compile cache on that GPU, and calibration
runs the warm-ups before the passes it compares (since 2026-11-25; acc6). The text pass of this task did not depend on
the cache in any run (the tester's GPUs reproduced `harness/ref_correct.json` exactly), unlike C15's.

## 6. Broken and cheating variants (the checks must reject them; ~8 min each)

```
bash harness/mkvariant.sh short && bash harness/run.sh v_short quick var_short
```
Variants: `short`, `lowres`, `skipimg`, `cheappatch`, `subpatch` (what each does: `variants/apply.py`). `quick` mode = warm-up, text passes,
image pass, one timed pair, teacher-forced re-read.

## 7. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | see `CARD_vllm27418.md` (REF and band are also in `harness/gate.json`) | verdict `ties_ref` |
| Calibration A/A | see card | within the card's A/A spread |
| Checks on the fix | counts (incl. image tokens), settings, teacher-forced re-read, text pass (bit-exact): all pass | no `fail:` |
| Broken variants | `short`, `lowres`, `skipimg`, `subpatch`: `fail:<checks>` as listed in the card. `cheappatch` (exact for still images, wrong only for video) is expected to pass, at the fix's speed | same checks fail; `cheappatch` `ties_ref` |

## Files

- `harness/run.sh` (job lists, STOP and busy-GPU checks, GPU lock), `harness/job.sh` (guarded run: busy-GPU refusal,
  foreign-process abort, VRAM watchdog, CPU pinning, clean environment), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/ref_correct.json` (the authors' reference text pass; a
  candidate's pass must equal this job's parent pass or this one), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/offline_c5y.py` (workload, text pass, image pass, teacher-forced re-read), `bench/run_bench.sh`; `loads/*.env` (official load `cal.env`,
  text pass `cor.env`, image pass `img.env`, re-read `tf.env`, transfer loads `tr_*.env`, `smoke.env`).
- `env/`: `build.env` (commits, wheel URLs and hashes, flashinfer sdist, triton-kernels commit, model revision), `deps.lock` (hashed), the authors' freezes and
  vllm file hashes. `setup/`: build, download, stage.
