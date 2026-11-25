# Run SGLang #20972 (no blocking copy when logprobs are returned) from this package

- PR: https://github.com/sgl-project/sglang/pull/20972, merged 2026-03-27.
- Human fix `a27651d5e0d138d2a5ddc529ee95317fd57328fa`; parent `e2b8463c80367730db57d5c38c8835ab326ab468`.
- Card: `CARD_sgl20972.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

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
cp -r sgl20972 /data/sgl20972 && cd /data/sgl20972     # any location; this is ISO_ROOT
cp site.env.example site.env                           # then edit: STAGE, GPU, CPUS, PORT (UV if uv is not on PATH)
```
Variables you set in the shell win over `site.env` (e.g. `GPU=1 CPUS=24-31,56-63 PORT=33500 bash harness/run.sh ...`).
Keep `$STAGE` short (< ~20 characters; IPC socket paths are limited to 107 characters) and on local disk, not NFS.

## 2. Build the two venvs (CPU and network; ~5-10 min)

```
bash setup/setup_lock.sh
```
Must end with `SETUP_LOCK_OK 20972`, after printing for both roles that the freeze equals the authors', that all sglang
files hash as in the authors' build, and the import check. Only `sglang/srt/layers/sampler.py` and
`sglang/srt/managers/scheduler_output_processor_mixin.py` differ between parent and fix.

## 3. Model and CUDA pieces (~2-5 min)

```
bash setup/get_model.sh      # Qwen/Qwen2.5-1.5B-Instruct @ 989aa798...; ends with GET_MODEL_OK
bash setup/fetch_cuda.sh     # CUDA 12.8 nvcc/cudart/cccl redist + the venv's cuBLAS/cuRAND/... headers; ends with FETCH_CUDA_OK
```

## 4. Stage to local disk (~2-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 20972
```

## 5. Calibration, then a scoring job of the human fix (one GPU; ~25 + ~10 min)

```
bash harness/run.sh cal_g0 cal          # 25 runs
bash harness/run.sh s1_g0 score         # 10 runs; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
A second GPU can run another job at the same time: `GPU=1 CPUS=... PORT=33500 bash harness/run.sh s2_g1 score`.
Transfer loads (batch 1, batch 32, top-20, no top-k): `bash harness/run.sh tr_g0 transfer`.

## 6. Broken variants (the checks must reject them; ~10 min each)

```
bash harness/mkvariant.sh droplp && bash harness/run.sh d1_g0 score var_droplp
```
Variants: `droplp`, `stalelp`, `fp16lp` (what each does: `variants/apply.py`).

## 7. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | REF 1.142 (7 jobs on 2 cards, 1.137-1.148) | speedup in **1.135-1.149** (REF ± 0.60 %), verdict `ties_ref` |
| Calibration | REF 1.142; A/A max deviation 0.58-0.65 % | A/A within ~1 % |
| Checks on the fix | counts, settings, exact digests (every rep), correctness (bit-identical): all pass | no `fail:` |
| Broken variants | `droplp`, `stalelp`, `fp16lp`: each fail:exact,correct (1.139-1.144x) | same |
| Transfer loads | batch 1 1.173x, batch 32 1.162x, top-20 1.157x, no top-k 1.002x | informational |
| GPU time | scoring job 9.4-10 min, calibration ~25 min, peak VRAM 20.6 GB | below the 97 % VRAM watchdog |

## Files

- `harness/run.sh` (job lists, STOP and busy-GPU checks, GPU lock), `harness/job.sh` (guarded run: busy-GPU refusal,
  foreign-process abort, VRAM watchdog, CPU pinning, clean environment), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/sgl_c5x.py` (workload, correctness pass; its teacher-forced mode is used by #16521 only), `bench/run_bench.sh`; `loads/*.env` (official
  load `cal.env`, correctness `cor.env`, transfer loads `tr_*.env`, `smoke.env`; `tf.env` is unused here).
- `env/`: `build.env` (commits, URLs, model revision), `deps.lock` (hashed), the authors' freezes, sglang file hashes,
  source hashes and `cuda_tree.sha256`. `setup/`: build, download, stage.
