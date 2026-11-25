# Run #6994 (SGLang, LoRA batch metadata without per-request syncs) from this package

- PR: https://github.com/sgl-project/sglang/pull/6994, merged 2025-06-11 (23:18 UTC).
- Human fix = merge commit `021f76e4f49861b2e9ea9ccff06a46d577e3c548`; parent `777688b8929c877e4e28c2eac208d776abe4c3af`.
- Card: `CARD_iso6994.md` (repo `raw/P2-port-4090/`). Environment details: `ENV.md`. Where each file comes from: `provenance/PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps below go from an empty Linux
machine to one calibration (REF and 5 A/A pairs) and one scoring job of the human fix, which is what the card reports.
There is no agent-facing harness for this task (no gate, vault or audit), so there is nothing to skip.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s) and a driver for CUDA 12.6 (torch 2.7.1+cu126).
  The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `flock`, `taskset` (util-linux), `nvidia-smi`, a host `python3` >= 3.8 (stdlib only).
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`, or `pip install uv==0.11.14`).
- Internet access to pypi.org, files.pythonhosted.org, codeload.github.com, huggingface.co.
- Disk: ~25 GB under `$ISO_ROOT` (two venvs and the uv cache, model 3.1 GB, LoRA 74 MB) and ~17 GB under `$STAGE`.

## 1. Settings (defaults in `env.sh`)

```
cp -r iso6994 /data/iso6994 && cd /data/iso6994   # any location; this is ISO_ROOT
export ISO_ROOT=$PWD                 # default: the directory holding env.sh
export STAGE=/tmp/iso6994            # default /tmp/iso6994; fast local disk (NVMe), not NFS
export UV=uv                         # default: uv on PATH
# optional: CPUS=0-7,32-39  (taskset list for the GPU jobs; default: the authors' CCD table on their pods, else no pinning)
# optional: PORT=31000      (base of the engine's ports; SGLang uses PORT+100; default 31000 + 200*GPU; one per GPU)
```
Keep `TMPDIR`-like paths short: `$STAGE` should be under ~20 characters (IPC socket paths are limited to 107 characters).

## 2. Build the two venvs (CPU and network; ~7 min, ~20 GB incl. uv cache)

```
bash b3/scripts/setup_lock.sh
```
Installs CPython 3.10.20 (uv-managed), then per role all dependencies from the hashed lock `env/6994_deps.lock`
(torch 2.7.1, sgl-kernel 0.1.7, triton 3.3.1, xgrammar 0.1.19, ...; flashinfer-python is left out, see `ENV.md`) and
SGLang's `python/` package from the commit's source tarball (codeload.github.com; every file checked against the
authors' hashes in `env/6994_<role>_src.sha256`, then built by its setuptools backend). It must end with
`SETUP_LOCK_OK 6994`, after printing for both roles that the freeze equals the authors' and that all SGLang files hash
exactly as in the authors' build. Only `sglang/srt/lora/lora_manager.py` and `sglang/srt/lora/mem_pool.py` differ
between parent and fix.

## 3. Download the model and make the LoRA adapters (~1-2 min, 3.1 GB)

```
bash b3/scripts/get_model.sh
```
`Qwen/Qwen2.5-1.5B-Instruct` at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` into `$ISO_ROOT/b3/hf` (LFS blobs
sha256-checked), then two random-weight rank-16 LoRA adapters (`b3/bench/make_lora.py`, seeds 0 and 1) into
`$ISO_ROOT/b3/lora/qwen15_r16{,b}`; their sha256 must equal the authors'. Must end with `GET_MODEL_OK`.

## 4. Stage to local disk (~4-15 min, ~17 GB; 15 min was measured from a busy NFS)

```
bash b3/scripts/pkg_stage.sh
```
Copies the interpreter, both venvs, the LoRA adapters and the model to `$STAGE` (file count and bytes checked). Must
end with `PKG_STAGE_OK`. Do not stage or build on a machine while timed jobs run on it.

## 5. One calibration and one scoring job on one GPU (~13.5 + ~4.5 min)

```
bash b3/scripts/b3_caldrv.sh 0 6994 cal:cal_g0 score:s1_g0       # GPU 0 (nvidia-smi index); runs the two jobs in order
```
- `cal`: 3 correctness passes (parent, parent, fix), 2 untimed warm-ups, 5 fix/parent pairs (AB/BA alternating) and
  5 parent/parent A/A pairs: 25 runs, ~13.5 min.
- `score`: 2 warm-ups, 1 correctness pass of the fix, 3 ABBA pairs: 9 runs, ~4.5 min. Add `:--transfer` to a score job
  (`score:s1_g0:--transfer`) for the card's three transfer loads (batch 16, batch 128, two adapters; +6 runs, ~4 min).
- Progress: `b3/logs/caldrv_<host>_g0.log` and `b3/cal/s6994/<label>/cal.log`. Stop cleanly between runs:
  `touch $ISO_ROOT/STOP_batch3`. A second GPU can run another job at the same time with its own `b3_caldrv.sh 1 ...`.

## 6. Read the result

```
python3 -I b3/scripts/cal_analyze.py b3/cal/s6994 b3/cal/s6994/cal_g0/cor_1_parent cal_g0 s1_g0
```
Per label: speedup of each pair (parent median rep / fix median rep), REF = median of the pairs, A/A pairs and their
max deviation, output check (token counts and output-text sha per rep vs the paired parent), correctness (fix vs the
first parent correctness pass), GPU job time. Expected, from the card:

| Quantity | Card | Pass if |
|---|---|---|
| Calibration REF | 1.475 (1.468-1.478) | in the band below |
| Scoring-job speedup | 1.474 (8 jobs, 1.465-1.485) | **1.458-1.490** (1.474 ± 1.1 %) |
| A/A max deviation | 1.15 % | ≲ 1.2 % |
| Correctness pass | fix vs parent 24/24 identical token ids | same |
| Timed outputs | token counts identical in every rep; text sha identical in only ~65-80 % of reps, even parent vs parent | counts identical; do not require identical sha (the card explains why) |
| Peak VRAM | 20.9 GB (0.85 of 24.5 GB) | below the 97 % watchdog |

Part of the between-job spread is a per-card offset (up to ~1.3 %, from the fix side): spread scoring jobs over cards.

## Files

- `b3/scripts/b3_caldrv.sh` (driver: lock per GPU, staging, STOP and busy-GPU checks), `b3_cal.sh` (the job list),
  `b3_job.sh` (guarded GPU run: busy-GPU refusal, foreign-process abort, VRAM watchdog, CPU pinning), `b3_stage.sh`,
  `b3_cpus.sh`, `hosttag.sh`, `cal_analyze.py`; `scripts/stage_tmp.sh` (interpreter staging); `screen/with_env.sh`.
- `b3/bench/sgl_b3.py` (the workload), `run_b3.sh`, `make_lora.py`; `b3/env/6994_{cal,cor,tr_b16,tr_b128,tr_2lora}.env`
  (the workload settings: official load, correctness pass, transfer loads).
- `env/`: build spec, lock, the authors' freezes and file hashes; `b3/scripts/setup_sgl.sh` is the
  authors' original resolver (date cutoff instead of a lock), kept as an alternative and as the lock's provenance.
