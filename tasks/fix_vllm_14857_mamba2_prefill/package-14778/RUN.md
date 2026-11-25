# Run #14778 (vLLM, Mamba2 prefill state updates with `torch.vmap`, later reverted) from this package

- PR: https://github.com/vllm-project/vllm/pull/14778, merged 2025-03-14 (20:36 UTC). It is the first landing of the change
  that #14857 re-landed; the two are close duplicates (count them as one mechanism).
- Human fix = merge commit `fe66b34728e5d383e3d19aefc544eeee808c99fb`; parent `270a5da495d24e947a71e2fa0c56635f4fad2dc3`.
- Card: `CARD_iso14778.md` (repo `raw/P2-port-4090/`). Environment details: `ENV.md`. Where each file comes from: `provenance/PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps below go from an empty Linux
machine to one calibration (REF and 5 A/A pairs) and one scoring job of the human fix, which is what the card reports.
There is no agent-facing harness for this task (no gate, vault or audit), so there is nothing to skip.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s) and a driver for CUDA 12.4 (torch 2.5.1+cu124).
  The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `flock`, `taskset` (util-linux), `nvidia-smi`, a host `python3` >= 3.8 (stdlib only).
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`, or `pip install uv==0.11.14`).
- Internet access to pypi.org, files.pythonhosted.org, wheels.vllm.ai, huggingface.co, github.com.
- Disk: ~22 GB under `$ISO_ROOT` (two venvs and the uv cache; the model is config + tokenizer only, ~10 MB) and ~15 GB under `$STAGE`.

## 1. Settings (defaults in `env.sh`)

```
cp -r iso14778 /data/iso14778 && cd /data/iso14778   # any location; this is ISO_ROOT
export ISO_ROOT=$PWD                 # default: the directory holding env.sh
export STAGE=/tmp/iso14778           # default /tmp/iso14778; fast local disk (NVMe), not NFS
export UV=uv                         # default: uv on PATH
# optional: CPUS=0-7,32-39  (taskset list for the GPU jobs; default: the authors' CCD table on their pods, else no pinning)
# optional: PORT=31000      (base of the engine's internal ports; default 31000 + 200*GPU; give each GPU its own)
```
Keep `TMPDIR`-like paths short: `$STAGE` should be under ~20 characters (vLLM's IPC socket path limit is 107).

## 2. Build the two venvs (CPU and network; ~8 min, ~22 GB incl. uv cache)

```
bash b3/scripts/setup_lock.sh
```
Installs CPython 3.10.20 (uv-managed), then per role all dependencies from the hashed lock `env/14778_deps.lock`
(torch 2.5.1, triton 3.1.0, transformers 4.49.0, ..., `xgrammar==0.1.17` instead of the deleted release this vLLM pins) and the per-commit vLLM
wheel from wheels.vllm.ai (sha256 checked). It must end with `SETUP_LOCK_OK 14778`, after printing for both roles that
the freeze equals the authors' and that all vLLM files hash exactly as in the authors' build. Between parent and fix
`vllm/model_executor/layers/mamba/mamba_mixer2.py`, `vllm/_version.py` and `vllm/_C.abi3.so` differ (the last is a
rebuild of unchanged sources).

## 3. Download the model config and tokenizer (< 1 min, ~10 MB)

```
bash b3/scripts/get_model.sh
```
`ibm-ai-platform/Bamba-9B` at revision `50ec8766310eae056e05e42ba70df45645912e1e` into `$ISO_ROOT/b3/hf`, without the
weights: this task loads **dummy weights** and only the first 16 of 32 layers (`load_format=dummy`,
`hf_overrides={"num_hidden_layers": 16}`; at this vLLM version the FP8 load used for #14857 crashes at model
construction, see the card). Must end with `GET_MODEL_OK`. The model is public (no token).

## 4. Stage to local disk (~3-4 min, ~15 GB)

```
bash b3/scripts/pkg_stage.sh
```
Copies the interpreter, both venvs and the model to `$STAGE` (file count and bytes checked). Must end with
`PKG_STAGE_OK`. Do not stage or build on a machine while timed jobs run on it: host stalls show up as slow parent reps.

## 5. One calibration and one scoring job on one GPU (~20.5 + ~7 min)

```
bash b3/scripts/b3_caldrv.sh 0 14778 cal:cal_g0 score:s1_g0      # GPU 0 (nvidia-smi index); runs the two jobs in order
```
- `cal`: 3 correctness passes (parent, parent, fix), 2 untimed warm-ups, 5 fix/parent pairs (AB/BA alternating) and
  5 parent/parent A/A pairs: 25 runs, ~20.5 min.
- `score`: 2 warm-ups, 1 correctness pass of the fix, 3 ABBA pairs: 9 runs, ~7 min. Add `:--transfer` to a score job
  (`score:s1_g0:--transfer`) for the card's three transfer loads (+6 runs, ~8 min).
- Progress: `b3/logs/caldrv_<host>_g0.log` and `b3/cal/s14778/<label>/cal.log`. Stop cleanly between runs:
  `touch $ISO_ROOT/STOP_batch3`. A second GPU can run another job at the same time with its own `b3_caldrv.sh 1 ...`.

## 6. Read the result

```
python3 -I b3/scripts/cal_analyze.py b3/cal/s14778 b3/cal/s14778/cal_g0/cor_1_parent cal_g0 s1_g0
```
Per label: speedup of each pair (parent median rep / fix median rep), REF = median of the pairs, A/A pairs and their
max deviation, output check (token counts and output sha per rep vs the paired parent), correctness (fix vs the first
parent correctness pass), GPU job time. Expected, from the card:

| Quantity | Card | Pass if |
|---|---|---|
| Calibration REF | 1.033 (main GPU1, 1.029-1.035); other cards read ~0.3 % higher | in the band below (GPU1 of the authors' main pod read low) |
| Scoring-job speedup | 1.0365 (5 jobs, 1.0334-1.0369) | **1.0313-1.0417** (1.0365 ± 0.5 %) |
| A/A max deviation | 0.31 % | ≲ 0.5 % |
| Outputs | identical token counts and sha in every rep; correctness 24/24 identical (dummy weights: bit-equality, not quality) | same |
| Peak VRAM | 21.2 GB (0.86 of the card) | below the 97 % watchdog |

Absolute times depend on the card (fix 3.80-3.91 s, parent 3.93-4.04 s per rep on 4090s); the ratio barely does.

## Files

- `b3/scripts/b3_caldrv.sh` (driver: lock per GPU, staging, STOP and busy-GPU checks), `b3_cal.sh` (the job list),
  `b3_job.sh` (guarded GPU run: busy-GPU refusal, foreign-process abort, VRAM watchdog, CPU pinning), `b3_stage.sh`,
  `b3_cpus.sh`, `hosttag.sh`, `cal_analyze.py`; `scripts/stage_tmp.sh` (interpreter staging); `screen/with_env.sh`.
- `b3/bench/offline_b3.py` (the workload) and `run_b3.sh`; `b3/env/14778_{cal,cor,tr_in256,tr_ol64,tr_seqs16}.env`
  (the workload settings: official load, correctness pass, transfer loads).
- `env/`: build spec, lock, the authors' freezes and file hashes; `b3/scripts/setup_b3.sh` + `screen/fetch.py` are the
  authors' original resolver (date cutoff instead of a lock), kept as an alternative and as the lock's provenance.
