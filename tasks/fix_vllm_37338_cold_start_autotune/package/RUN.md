# Run vLLM #37338 (Qwen3.5 cold start: GDN warm-up with the right dtypes) from this package

- PR: https://github.com/vllm-project/vllm/pull/37338, merged 2026-03-23.
- Human fix `a16133a0f12e8e406206be1bc8041727d9f1ea7d`; parent `54ab804e87dd534c0a8cb8ad8eec7ad5d75775ae`.
- Card: `CARD_vllm37338.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps go from an empty Linux machine
to one calibration and one scoring job of the human fix (what the card reports), plus the broken and cheating variants
that the checks must reject. This is a **cold-start** task: every timed run starts with empty compile caches, and the
score is the time from launching the Python process to the first `generate()` result.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s, driver 550 = CUDA 12.4; torch 2.10.0+cu128 runs on
  it). The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `flock`, `taskset`, `nvidia-smi`, a host `python3` >= 3.8 (stdlib only). No CUDA toolkit.
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`).
- Internet access to pypi.org, files.pythonhosted.org, wheels.vllm.ai, huggingface.co.
- Disk: ~35 GB under `$ISO_ROOT` (two venvs ~9 GB each, uv cache, model 8.8 GB), ~30 GB under `$STAGE` (+ ~0.4 GB of
  compile caches per running job, deleted after each run).

## 1. Settings

```
cp -r vllm37338 /data/vllm37338 && cd /data/vllm37338   # any location; this is ISO_ROOT
cp site.env.example site.env                            # then edit: STAGE, GPU, CPUS, PORT (UV if uv is not on PATH)
```
Variables you set in the shell win over `site.env` (e.g. `GPU=1 CPUS=8-15,40-47 PORT=36500 bash harness/run.sh ...`).
**CPU sets are host-specific**: pick the cores near the GPU (`nvidia-smi topo -m`, column "CPU Affinity"). On the
authors' hosts: main pod GPU0 `0-7,32-39`, GPU1 `8-15,40-47`, GPU2 `16-23,48-55`, GPU3 `24-31,56-63`; n1 pod
GPU1 `24-31,56-63` (check the others with `nvidia-smi topo -m`). Omitting `CPUS` runs unpinned (noisier).
Keep `$STAGE` short (< ~20 characters; IPC socket paths are limited to 107 characters) and on local disk, not NFS.
Start-up compiles on the CPU, so give each job its own CPU set (one CCD + SMT siblings on the authors' host).

## 2. Build the two venvs (CPU and network; ~5-10 min)

```
bash setup/setup_lock.sh
```
Must end with `SETUP_LOCK_OK 37338`, after printing for both roles that the freeze equals the authors', that all vllm
files hash as in the authors' build, and the import check.

## 3. Model (~3-6 min)

```
bash setup/get_model.sh      # Qwen/Qwen3.5-4B @ 851bf6e8...; ends with GET_MODEL_OK
```

## 4. Stage to local disk (~3-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 37338
```

## 5. Calibration, then a scoring job of the human fix (one GPU; ~80 + ~25 min)

```
bash harness/run.sh cal_g0 cal          # 22 cold runs (~3 min each) + the teacher-forced check
bash harness/run.sh s1_g0 score         # 7 cold runs + the teacher-forced check; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
A second GPU can run another job at the same time: `GPU=1 CPUS=<cores near GPU1> PORT=36500 bash harness/run.sh s2_g1 score`.
Transfer loads (first batch 1 x 128 / 16 x 2048 ids; warm caches): `bash harness/run.sh tr_g0 transfer`.
To re-derive a verdict with other thresholds: edit `harness/gate.json`, then `python3 -I harness/check.py $ISO_ROOT <label>`.

## 6. Broken and cheating variants (the checks must reject them; ~10 min each)

```
bash harness/run.sh donor1 donor                     # once: one parent run whose caches the shipcache variant ships
bash harness/mkvariant.sh shipcache && bash harness/run.sh v_shipcache quick var_shipcache
```
Variants: `shipcache`, `persist`, `persisttmp`, `short1`, `noautotune`, `gbf16` (what each does: `variants/apply.py`).
`quick` mode = one cold pair (candidate, parent) + teacher-forced check. For `persist`, run `quick` twice on the same
variant venv (`v_persist`, `v_persist2`): the second job starts with the cache the first one left inside the venv.

## 7. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | see `CARD_vllm37338.md` (REF and band are also in `harness/gate.json`) | verdict `ties_ref` |
| Calibration A/A | see card | within the card's A/A spread |
| Checks on the fix | counts, settings, teacher-forced, audit, venv, cold witness, steady, after: all pass | no `fail:` |
| Variants | each `fail:<checks>` as listed in the card | same checks fail |

## Files

- `harness/run.sh` (job lists, fresh caches per run, cold witness, venv snapshot, STOP and busy-GPU checks, GPU lock),
  `harness/job.sh` (guarded run: busy-GPU refusal, foreign-process abort, VRAM watchdog, CPU pinning, clean
  environment), `harness/audit.py` (static audit of the candidate venv), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/offline_c5y.py` (workload with wall-clock stamps, settings readback, teacher-forced check),
  `bench/run_bench.sh`; `loads/*.env` (official load `cal.env`, teacher-forced `tf.env`, transfers `tr_*.env`, `smoke.env`).
- `env/`: `build.env` (commits, wheel URLs and sha256, model revision), `deps.lock` (hashed), the authors' freezes and
  vllm file hashes. `setup/`: build, download, stage.
