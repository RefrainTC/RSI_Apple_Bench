# Run SGLang #16521 (radix-cache key includes the generated tokens) from this package

- PR: https://github.com/sgl-project/sglang/pull/16521, merged 2026-02-10.
- Human fix `2bfab1bb67460af8b125d94ca4c5ab0040c12cd2`; parent `4262f5259b94c5c08779efff5017c0a6235bcb5f`.
- Card: `CARD_sgl16521.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps go from an empty Linux machine
to one calibration and one scoring job of the human fix (what the card reports), plus the broken variants that the checks
must reject.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s, driver 550 = CUDA 12.4; torch 2.9.1+cu128 runs on it).
  The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `xz`, `flock`, `taskset`, `nvidia-smi`, `g++` (FlashInfer compiles one module at engine start;
  the authors had gcc 12.2), a host `python3` >= 3.8 (stdlib only).
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`).
- Internet access to pypi.org, files.pythonhosted.org, codeload.github.com, huggingface.co, developer.download.nvidia.com.
- Disk: ~35 GB under `$ISO_ROOT` (two venvs ~11 GB each, uv cache, model 3.1 GB, CUDA pieces 0.35 GB), ~30 GB under `$STAGE`.

## 1. Settings

```
cp -r sgl16521 /data/sgl16521 && cd /data/sgl16521     # any location; this is ISO_ROOT
cp site.env.example site.env                           # then edit: STAGE, GPU, CPUS, PORT (UV if uv is not on PATH)
```
Variables you set in the shell win over `site.env` (e.g. `GPU=1 CPUS=24-31,56-63 PORT=33500 bash harness/run.sh ...`).
Keep `$STAGE` short (< ~20 characters; IPC socket paths are limited to 107 characters) and on local disk, not NFS.

## 2. Build the two venvs (CPU and network; ~5-10 min)

```
bash setup/setup_lock.sh
```
Must end with `SETUP_LOCK_OK 16521`, after printing for both roles that the freeze equals the authors', that all sglang
files hash as in the authors' build, and the import check. Only `sglang/srt/mem_cache/radix_cache.py` (and
`_version.py`) differ between parent and fix.

## 3. Model and CUDA pieces (~2-5 min)

```
bash setup/get_model.sh      # Qwen/Qwen2.5-1.5B-Instruct @ 989aa798...; ends with GET_MODEL_OK
bash setup/fetch_cuda.sh     # CUDA 12.8 nvcc/cudart/cccl redist + the venv's cuBLAS/cuRAND/... headers; ends with FETCH_CUDA_OK
```

## 4. Stage to local disk (~2-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 16521
```

## 5. Calibration, then a scoring job of the human fix (one GPU; ~25 + ~10 min)

```
bash harness/run.sh cal_g0 cal          # 25 runs + the teacher-forced check
bash harness/run.sh s1_g0 score         # 10 runs + the teacher-forced check; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
A second GPU can run another job at the same time: `GPU=1 CPUS=... PORT=33500 bash harness/run.sh s2_g1 score`.
Transfer loads (turn-1 output 64 / 1024 tokens, page size 16): `bash harness/run.sh tr_g0 transfer`.

## 6. Broken variants (the checks must reject them; ~10 min each)

```
bash harness/mkvariant.sh drophist && bash harness/run.sh k2_g0 score var_drophist
```
Variants: `drophist`, `fp8kv`, `short2`, `kvzero`, `kvnoise100`, `kvnoise10`, `kvnoise1`, `kvperm` (what each does:
`variants/apply.py`; expected verdicts below).

## 7. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | REF 3.827 (8 jobs on 2 cards, 3.794-3.871) | speedup in **3.779-3.875** (REF ± 1.26 %), verdict `ties_ref` |
| Calibration | REF 3.833; A/A max deviation 0.27-0.61 % | A/A within ~1 % |
| Checks on the fix | counts, settings, correctness, teacher-forced (max gap 0.125 nats, <= 0.34 % of tokens below the top-1): all pass | no `fail:` |
| Turn-2 outputs vs parent | 243/256 requests identical per pair (deterministic) | informational |
| Broken variants | `drophist` fail:tf (4.28x); `fp8kv` fail:settings,tf; `short2` fail:counts (4.86x); `kvzero` fail:tf; `kvnoise100` fail:tf (correctness pass); `kvperm`, `kvnoise10`, `kvnoise1` pass (their outputs are as correct as the fix's) | same |
| GPU time | scoring job 8.6-8.9 min, calibration 22 min, peak VRAM 20.4 GB | below the 97 % VRAM watchdog |

Part of the between-job spread is a ~1 % card offset (n1 GPU1 reads lower than GPU0): spread scoring jobs over cards.

## Files

- `harness/run.sh` (job lists, STOP and busy-GPU checks, GPU lock), `harness/job.sh` (guarded run: busy-GPU refusal,
  foreign-process abort, VRAM watchdog, CPU pinning, clean environment), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/sgl_c5x.py` (workload, correctness pass, teacher-forced check), `bench/run_bench.sh`; `loads/*.env` (official
  load `cal.env`, correctness `cor.env`, teacher-forced `tf.env`, transfer loads `tr_*.env`, `smoke.env`).
- `env/`: `build.env` (commits, URLs, model revision), `deps.lock` (hashed), the authors' freezes, sglang file hashes,
  source hashes and `cuda_tree.sha256`. `setup/`: build, download, stage.
