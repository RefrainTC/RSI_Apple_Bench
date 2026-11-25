# Run SGLang #17005 (tokenizer manager: logprob lists are extended with the new positions instead of rebuilt on every message) from this package

- PR: https://github.com/sgl-project/sglang/pull/17005, merged 2026-01-13.
- Human fix `339915ce2b8c9c6cb73a4a52d3d084d8f66c967f`; parent `075c5a5789e77de393dc5490099a8f7d3ccd4d12`.
- Card: `CARD_sgl17005.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps go from an empty Linux machine
to one calibration and one scoring job of the human fix (what the card reports), plus the broken variants that the
checks must reject.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s, driver 550 = CUDA 12.4; torch 2.9.1 is the CUDA 12.8
  build and runs on it). The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `xz`, `flock`, `taskset`, `nvidia-smi`, a host `python3` >= 3.8 (stdlib only), and a host C++
  compiler (`g++`; the authors' pods have gcc 12.2), in case FlashInfer JIT-compiles a module at engine start with the
  CUDA 12.8 `nvcc` that `setup/fetch_cuda.sh` downloads (the screen ran with the same tree). No system CUDA toolkit is
  needed.
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`).
- Internet access to pypi.org, files.pythonhosted.org, codeload.github.com, developer.download.nvidia.com, huggingface.co.
- Disk: ~35 GB under `$ISO_ROOT` (two venvs ~10 GB each, uv cache, CUDA tree, model 1 GB), ~25 GB under `$STAGE`.

## 1. Settings

```
cp -r sgl17005 /data/sgl17005 && cd /data/sgl17005     # any location; this is ISO_ROOT
cp site.env.example site.env                           # then edit: STAGE, GPU, CPUS, PORT (UV if uv is not on PATH)
```
Variables you set in the shell win over `site.env` (e.g. `GPU=1 CPUS=8-15,40-47 PORT=36500 bash harness/run.sh ...`).
**CPU sets are host-specific**: pick the cores near the GPU (`nvidia-smi topo -m`, column "CPU Affinity"). On the
authors' hosts: main pod GPU0 `0-7,32-39`, GPU1 `8-15,40-47`, GPU2 `16-23,48-55`, GPU3 `24-31,56-63`; n1 pod
GPU1 `24-31,56-63` (check the others with `nvidia-smi topo -m`). Omitting `CPUS` runs unpinned (noisier).
Keep `$STAGE` short (< ~20 characters; IPC socket paths are limited to 107 characters) and on local disk, not NFS.

## 2. Build the two venvs (CPU and network; ~5-20 min)

```
bash setup/setup_lock.sh
```
Must end with `SETUP_LOCK_OK 17005`, after printing for both roles that the source files hash as the authors', that the
freeze equals the authors', that all sglang files hash as in the authors' build, and the import check. Only
`sglang/srt/managers/tokenizer_manager.py` (and `_version.py`) differs between parent and fix.

## 3. CUDA pieces for FlashInfer's JIT (~2-5 min; the same tree as the screen's)

```
bash setup/fetch_cuda.sh     # ends with FETCH_CUDA_OK; the tree must hash as env/cuda_tree.sha256
```

## 4. Model (~5-10 min)

```
bash setup/get_model.sh      # Qwen/Qwen2.5-0.5B-Instruct @ 7ae55760...; ends with GET_MODEL_OK
```

## 5. Stage to local disk (~2-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 17005
```

## 6. Calibration, then a scoring job of the human fix (one GPU; ~80 + ~32 min)

```
bash harness/run.sh cal_g0 cal          # 25 runs
bash harness/run.sh s1_g0 score         # 10 runs; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
Every run sets `SGLANG_DISABLE_CUDNN_CHECK=1` (`harness/run.sh`): this SGLang refuses to start on torch 2.9.1 with
cuDNN 9.10 because of a Conv3d bug; this model has no convolution.
A second GPU can run another job at the same time: `GPU=1 CPUS=<cores near GPU1> PORT=36500 bash harness/run.sh s2_g1 score`.
Transfer loads (top 5 logprobs instead of 20; no logprobs at all, a negative control): `bash harness/run.sh tr_g0 transfer`.
To re-derive a verdict with other thresholds: edit `harness/gate.json`, then `python3 -I harness/check.py $ISO_ROOT <label>`.

## 7. Broken variants (the checks must reject them; ~13 min each)

```
bash harness/mkvariant.sh short && bash harness/run.sh v_short quick var_short
```
Variants: `short`, `stale`, `notop` (what each does: `variants/apply.py`). `quick` mode = warm-up, correctness
passes, one timed pair.

## 8. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | see `CARD_sgl17005.md` (REF and band are also in `harness/gate.json`) | verdict `ties_ref` |
| Calibration A/A | see card | within the card's A/A spread |
| Checks on the fix | counts, settings, exact timed outputs and logprobs, correctness pass with top-5 lists (bit-exact): all pass | no `fail:` |
| Broken variants | each `fail:<checks>` as listed in the card | same verdicts |

## Files

- `harness/run.sh` (job lists, STOP and busy-GPU checks, GPU lock), `harness/job.sh` (guarded run: busy-GPU refusal,
  foreign-process abort, VRAM watchdog, CPU pinning, clean environment), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/ref_correct.json` (the authors' reference correctness pass; a
  candidate's pass must equal this job's parent pass or this one), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/sgl_c5y.py` (workload, correctness pass), `bench/run_bench.sh`; `loads/*.env` (official load `cal.env`,
  correctness `cor.env`, transfer loads `tr_*.env`, `smoke.env`).
- `env/`: `build.env` (commits, URLs, model revision), `deps.lock` (hashed), `cuda_headers.lock`, `cuda_tree.sha256`, the
  authors' freezes, sglang file hashes and source hashes. `setup/`: build, CUDA pieces, download, stage.
