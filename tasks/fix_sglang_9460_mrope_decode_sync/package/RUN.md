# Run SGLang #9460 (M-RoPE decode positions without per-request syncs) from this package

- PR: https://github.com/sgl-project/sglang/pull/9460, merged 2025-09-03.
- Human fix `4ed9053ecffb910c87d6c0db4ef64a7c54aa099b`; parent `5e19b159b02ae1b519ce6c3a0d1e7ff04ab9fcd5`.
- Card: `CARD_sgl9460.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps go from an empty Linux machine
to one calibration and one scoring job of the human fix (what the card reports), plus the broken variants that the
checks must reject.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s, driver 550 = CUDA 12.4; torch 2.8.0+cu128 runs on it).
  The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `flock`, `taskset`, `nvidia-smi`, a host `python3` >= 3.8 (stdlib only). No CUDA toolkit and no
  compiler are needed (Triton ships its own `ptxas`).
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`).
- Internet access to pypi.org, files.pythonhosted.org, codeload.github.com, huggingface.co.
- Disk: ~30 GB under `$ISO_ROOT` (two venvs ~8.5 GB each, uv cache, model 4.2 GB), ~25 GB under `$STAGE`.

## 1. Settings

```
cp -r sgl9460 /data/sgl9460 && cd /data/sgl9460       # any location; this is ISO_ROOT
cp site.env.example site.env                          # then edit: STAGE, GPU, CPUS, PORT (UV if uv is not on PATH)
```
Variables you set in the shell win over `site.env` (e.g. `GPU=1 CPUS=8-15,40-47 PORT=36500 bash harness/run.sh ...`).
**CPU sets are host-specific**: pick the cores near the GPU (`nvidia-smi topo -m`, column "CPU Affinity"). On the
authors' hosts: main pod GPU0 `0-7,32-39`, GPU1 `8-15,40-47`, GPU2 `16-23,48-55`, GPU3 `24-31,56-63`; n1 pod
GPU1 `24-31,56-63` (check the others with `nvidia-smi topo -m`). Omitting `CPUS` runs unpinned (noisier).
Keep `$STAGE` short (< ~20 characters; IPC socket paths are limited to 107 characters) and on local disk, not NFS.

## 2. Build the two venvs (CPU and network; ~5-10 min)

```
bash setup/setup_lock.sh
```
Must end with `SETUP_LOCK_OK 9460`, after printing for both roles that the freeze equals the authors', that all sglang
files hash as in the authors' build, and the import check. Only `sglang/srt/model_executor/forward_batch_info.py` and
`sglang/srt/layers/rotary_embedding.py` differ between parent and fix.

## 3. Model (~2-5 min)

```
bash setup/get_model.sh      # Qwen/Qwen2-VL-2B-Instruct @ 1860c46f...; ends with GET_MODEL_OK
```

## 4. Stage to local disk (~2-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 9460
```

## 5. Calibration, then a scoring job of the human fix (one GPU; ~25 + ~10 min)

```
bash harness/run.sh cal_g0 cal          # 25 runs + the teacher-forced check
bash harness/run.sh s1_g0 score         # 10 runs + the teacher-forced check; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
A second GPU can run another job at the same time: `GPU=1 CPUS=<cores near GPU1> PORT=36500 bash harness/run.sh s2_g1 score`.
Transfer loads (32 requests per rep; text-only prompts): `bash harness/run.sh tr_g0 transfer`.
To re-derive a verdict with other thresholds: edit `harness/gate.json`, then `python3 -I harness/check.py $ISO_ROOT <label>`.

## 6. Broken variants (the checks must reject them; ~5 min each)

```
bash harness/mkvariant.sh nodelta && bash harness/run.sh v_nodelta quick var_nodelta
```
Variants: `nodelta`, `offby1`, `frozen`, `fp8kv`, `short`, `lowres` (what each does: `variants/apply.py`). `quick` mode =
warm-up, correctness passes, one timed pair, teacher-forced check.

## 7. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | see `CARD_sgl9460.md` (REF and band are also in `harness/gate.json`) | verdict `ties_ref` |
| Calibration A/A | see card | within the card's A/A spread |
| Checks on the fix | counts, settings, correctness pass (bit-exact), teacher-forced: all pass | no `fail:` |
| Broken variants | every one `fail:<checks>` as listed in the card | same checks fail |

## Files

- `harness/run.sh` (job lists, STOP and busy-GPU checks, GPU lock), `harness/job.sh` (guarded run: busy-GPU refusal,
  foreign-process abort, VRAM watchdog, CPU pinning, clean environment), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/ref_correct.json` (the authors' reference correctness pass; a
  candidate's pass must equal this job's parent pass or this one), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/sgl_c5y.py` (workload, correctness pass, teacher-forced check with HF transformers), `bench/run_bench.sh`;
  `loads/*.env` (official load `cal.env`, correctness `cor.env`, teacher-forced `tf.env`, transfer loads `tr_*.env`,
  `smoke.env`).
- `env/`: `build.env` (commits, URLs, model revision), `deps.lock` (hashed), the authors' freezes, sglang file hashes and
  source hashes. `setup/`: build, download, stage.
