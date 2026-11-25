# Run SGLang #27868 (Qwen3.5 decode: GDN in-projections and Q/K norm on a second CUDA stream) from this package

- PR: https://github.com/sgl-project/sglang/pull/27868, merged 2026-06-15.
- Human fix `d5899b95c4d54db58d5412b9ce0cc487b212d9a0`; parent `09e9c4fde3b4326c4a66fbb014d9c7a6b8995b30`.
- Card: `CARD_sgl27868.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps go from an empty Linux machine
to one calibration and one scoring job of the human fix (what the card reports), plus the broken variants that the
checks must reject.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s, driver 550 = CUDA 12.4; the cu129 torch wheels run on
  it). The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `xz`, `flock`, `taskset`, `nvidia-smi`, a host `python3` >= 3.8 (stdlib only), and a host C++
  compiler (`g++`; the authors' pods have gcc 12.2): this SGLang JIT-compiles FlashInfer's norm module at engine start
  with the CUDA 12.8 `nvcc` that `setup/fetch_cuda.sh` downloads. No system CUDA toolkit is needed.
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`).
- Internet access to pypi.org, files.pythonhosted.org, download.pytorch.org, download-r2.pytorch.org, github.com
  (sglang-kernel wheel), codeload.github.com, developer.download.nvidia.com, huggingface.co.
- Disk: ~40 GB under `$ISO_ROOT` (two venvs ~11 GB each, uv cache, CUDA tree, model 9 GB), ~35 GB under `$STAGE`.

## 1. Settings

```
cp -r sgl27868 /data/sgl27868 && cd /data/sgl27868     # any location; this is ISO_ROOT
cp site.env.example site.env                           # then edit: STAGE, GPU, CPUS, PORT (UV if uv is not on PATH)
```
Variables you set in the shell win over `site.env` (e.g. `GPU=1 CPUS=8-15,40-47 PORT=36500 bash harness/run.sh ...`).
**CPU sets are host-specific**: pick the cores near the GPU (`nvidia-smi topo -m`, column "CPU Affinity"). On the
authors' hosts: main pod GPU0 `0-7,32-39`, GPU1 `8-15,40-47`, GPU2 `16-23,48-55`, GPU3 `24-31,56-63`; n1 pod
GPU1 `24-31,56-63` (check the others with `nvidia-smi topo -m`). Omitting `CPUS` runs unpinned (noisier).
Keep `$STAGE` short (< ~20 characters; IPC socket paths are limited to 107 characters) and on local disk, not NFS.

## 2. Build the two venvs (CPU and network; ~15-50 min)

```
bash setup/setup_lock.sh
```
Must end with `SETUP_LOCK_OK 27868`, after printing for both roles that the source files hash as the authors', that the
freeze equals the authors', that all sglang files hash as in the authors' build, and the import check. Only
`sglang/srt/models/qwen3_5.py` differs between parent and fix. The script drops the Rust extension block from
`pyproject.toml` (identically in both roles) and uninstalls `nvidia-cutlass-dsl-libs-cu13` after the lock (`ENV.md`).

## 3. CUDA pieces for FlashInfer's JIT (~2-5 min)

```
bash setup/fetch_cuda.sh     # ends with FETCH_CUDA_OK; the tree must hash as env/cuda_tree.sha256
```

## 4. Model (~5-10 min)

```
bash setup/get_model.sh      # Qwen/Qwen3.5-4B @ 851bf6e8...; ends with GET_MODEL_OK
```

## 5. Stage to local disk (~2-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 27868
```

## 6. Calibration, then a scoring job of the human fix (one GPU; ~25 + ~10 min)

```
bash harness/run.sh cal_g0 cal          # 25 runs
bash harness/run.sh s1_g0 score         # 10 runs; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
A second GPU can run another job at the same time: `GPU=1 CPUS=<cores near GPU1> PORT=36500 bash harness/run.sh s2_g1 score`.
Transfer loads (1 and 8 requests per rep): `bash harness/run.sh tr_g0 transfer`.
To re-derive a verdict with other thresholds: edit `harness/gate.json`, then `python3 -I harness/check.py $ISO_ROOT <label>`.

## 7. Broken variants (the checks must reject them; ~5 min each)

```
bash harness/mkvariant.sh short && bash harness/run.sh v_short quick var_short
```
Variants: `short`, `fp8kv`, `noknorm` (what each does: `variants/apply.py`). `quick` mode = warm-up, correctness
passes, one timed pair.

## 8. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | see `CARD_sgl27868.md` (REF and band are also in `harness/gate.json`) | verdict `ties_ref` |
| Calibration A/A | see card | within the card's A/A spread |
| Checks on the fix | counts, settings, exact timed outputs, correctness pass (bit-exact): all pass | no `fail:` |
| Broken variants | every one `fail:<checks>` as listed in the card | same checks fail |

## Files

- `harness/run.sh` (job lists, STOP and busy-GPU checks, GPU lock), `harness/job.sh` (guarded run: busy-GPU refusal,
  foreign-process abort, VRAM watchdog, CPU pinning, clean environment), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/ref_correct.json` (the authors' reference correctness pass; a
  candidate's pass must equal this job's parent pass or this one), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/sgl_c5y.py` (workload, correctness pass), `bench/run_bench.sh`; `loads/*.env` (official load `cal.env`,
  correctness `cor.env`, transfer loads `tr_*.env`, `smoke.env`).
- `env/`: `build.env` (commits, URLs, model revision), `deps.lock` (hashed), `cuda_headers.lock`, `cuda_tree.sha256`, the
  authors' freezes, sglang file hashes and source hashes. `setup/`: build, CUDA pieces, download, stage.
