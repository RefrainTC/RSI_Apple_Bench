# Run #14857 (vLLM, Mamba2 prefill without per-sequence syncs) from this package

- PR: https://github.com/vllm-project/vllm/pull/14857, merged 2025-03-21 (02:21 UTC).
- Human fix = merge commit `296f927f2493908984707354e3cc5d7b2e41650b`; parent `0032903a5bb7c7c655f52f4efdfcc221947e9ca8`.
- Card: `CARD_iso14857.md` (repo `raw/P2-port-4090/`). Environment details: `ENV.md`. Where each file comes from: `provenance/PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps below go from an empty Linux
machine to one calibration (REF and 5 A/A pairs) and one scoring job of the human fix, which is what the card reports.
There is no agent-facing harness for this task (no gate, vault or audit), so there is nothing to skip.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s) and a driver for CUDA 12.4 (torch 2.6.0+cu124).
  The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `flock`, `taskset` (util-linux), `nvidia-smi`, a host `python3` >= 3.8 (stdlib only).
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`, or `pip install uv==0.11.14`).
- Internet access to pypi.org, files.pythonhosted.org, wheels.vllm.ai, huggingface.co, github.com.
- Disk: ~42 GB under `$ISO_ROOT` (two venvs ~11 GB each with the uv cache, model 19.6 GB) and ~33 GB under `$STAGE`.

## 1. Settings (defaults in `env.sh`)

```
cp -r iso14857 /data/iso14857 && cd /data/iso14857   # any location; this is ISO_ROOT
export ISO_ROOT=$PWD                 # default: the directory holding env.sh
export STAGE=/tmp/iso14857           # default /tmp/iso14857; fast local disk (NVMe), not NFS
export UV=uv                         # default: uv on PATH
# optional: CPUS=0-7,32-39  (taskset list for the GPU jobs; default: the authors' CCD table on their pods, else no pinning)
# optional: PORT=31000      (base of the engine's internal ports; default 31000 + 200*GPU; give each GPU its own)
```
Keep `TMPDIR`-like paths short: `$STAGE` should be under ~20 characters (vLLM's IPC socket path limit is 107).

## 2. Build the two venvs (CPU and network; ~8 min, ~22 GB incl. uv cache)

```
bash b3/scripts/setup_lock.sh
```
Installs CPython 3.10.20 (uv-managed), then per role all dependencies from the hashed lock `env/14857_deps.lock`
(torch 2.6.0, triton 3.2.0, transformers, ..., `xgrammar==0.1.17` instead of the deleted 0.1.16) and the per-commit vLLM
wheel from wheels.vllm.ai (sha256 checked). It must end with `SETUP_LOCK_OK 14857`, after printing for both roles that
the freeze equals the authors' and that all vLLM files hash exactly as in the authors' build. Only
`vllm/_version.py` and `vllm/model_executor/layers/mamba/mamba_mixer2.py` differ between parent and fix.

## 3. Download the model (~3-5 min, 19.6 GB)

```
bash b3/scripts/get_model.sh
```
`ibm-ai-platform/Bamba-9B` at revision `50ec8766310eae056e05e42ba70df45645912e1e` into `$ISO_ROOT/b3/hf`; every LFS
blob is checked against its sha256. Must end with `GET_MODEL_OK`. The model is public (no token).

## 4. Stage to local disk (~7-18 min, ~33 GB; 18 min was measured from a busy NFS)

```
bash b3/scripts/pkg_stage.sh
```
Copies the interpreter, both venvs and the model to `$STAGE` (file count and bytes checked). Must end with
`PKG_STAGE_OK`. Do not stage or build on a machine while timed jobs run on it: host stalls show up as slow parent reps.

## 5. One calibration and one scoring job on one GPU (~25 + ~10 min)

```
bash b3/scripts/b3_caldrv.sh 0 14857 cal:cal_g0 score:s1_g0      # GPU 0 (nvidia-smi index); runs the two jobs in order
```
- `cal`: 3 correctness passes (parent, parent, fix), 2 untimed warm-ups, 5 fix/parent pairs (AB/BA alternating) and
  5 parent/parent A/A pairs: 25 runs, ~24.5 min.
- `score`: 2 warm-ups, 1 correctness pass of the fix, 3 ABBA pairs: 9 runs, ~9.5 min. Add `:--transfer` to a score job
  (`score:s1_g0:--transfer`) for the card's three transfer loads (+6 runs, ~9 min).
- Progress: `b3/logs/caldrv_<host>_g0.log` and `b3/cal/s14857/<label>/cal.log`. Stop cleanly between runs:
  `touch $ISO_ROOT/STOP_batch3`. A second GPU can run another job at the same time with its own `b3_caldrv.sh 1 ...`.

## 6. Read the result

```
python3 -I b3/scripts/cal_analyze.py b3/cal/s14857 b3/cal/s14857/cal_g0/cor_1_parent cal_g0 s1_g0
```
Per label: speedup of each pair (parent median rep / fix median rep), REF = median of the pairs, A/A pairs and their
max deviation, output check (token counts and output sha per rep vs the paired parent), correctness (fix vs the first
parent correctness pass), GPU job time. Expected, from the card:

| Quantity | Card | Pass if |
|---|---|---|
| Calibration REF | 1.063 (1.0625-1.0664) | in the band below |
| Scoring-job speedup | 1.0648 (7 jobs, 1.0614-1.0656) | **1.0595-1.0701** (1.0648 ± 0.5 %) |
| A/A max deviation | 0.33 % | ≲ 0.5 % (a single host stall can make one pair ~1.1-1.3; the median of pairs absorbs it) |
| Outputs | identical token counts and sha in every rep; correctness 24/24 identical | same |
| Peak VRAM | 22.1 GB (90 % of 24 GB; vLLM loads bf16 then quantises to FP8) | below the 97 % watchdog |

Absolute times depend on the card (fix 5.33-5.45 s, parent 5.69-5.80 s per rep on 4090s); the ratio does not.

## Files

- `b3/scripts/b3_caldrv.sh` (driver: lock per GPU, staging, STOP and busy-GPU checks), `b3_cal.sh` (the job list),
  `b3_job.sh` (guarded GPU run: busy-GPU refusal, foreign-process abort, VRAM watchdog, CPU pinning), `b3_stage.sh`,
  `b3_cpus.sh`, `hosttag.sh`, `cal_analyze.py`; `scripts/stage_tmp.sh` (interpreter staging); `screen/with_env.sh`.
- `b3/bench/offline_b3.py` (the workload) and `run_b3.sh`; `b3/env/14857_{cal,cor,tr_in256,tr_ol64,tr_seqs16}.env`
  (the workload settings: official load, correctness pass, transfer loads).
- `env/`: build spec, lock, the authors' freezes and file hashes; `b3/scripts/setup_b3.sh` + `screen/fetch.py` are the
  authors' original resolver (date cutoff instead of a lock), kept as an alternative and as the lock's provenance.
