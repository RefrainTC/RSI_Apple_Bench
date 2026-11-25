# Run vLLM #30056 (structured output with a reasoning parser: end-of-reasoning check on the new tokens only) from this package

- PR: https://github.com/vllm-project/vllm/pull/30056, merged 2025-12-09.
- Human fix `c72ea1072313b30fc8b4701697520fdd0b56fc35`; parent `67475a6e81abea915857f82e6f10d80b03b842c9`.
- Card: `CARD_vllm30056.md` (repo `raw/P2-port-4090/`). Environment: `ENV.md`. Where each file comes from: `PROVENANCE.md`.

This directory is self-contained. Copy it anywhere; that copy is `$ISO_ROOT`. The steps go from an empty Linux machine
to one calibration and one scoring job of the human fix (what the card reports), plus the broken variants that the
checks must reject.

## 0. What you need

- Linux x86_64, one NVIDIA GPU with 24 GB (the card used RTX 4090s, driver 550 = CUDA 12.4; torch 2.9.0 (CUDA 12.8
  build) runs on it). The GPU must be idle: every run refuses to start, and aborts, if another process is on it.
- `bash`, `curl`, `tar`, `flock`, `taskset`, `nvidia-smi`, a host `python3` >= 3.8 (stdlib only). No CUDA toolkit.
- uv 0.11.14 (`curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`).
- Internet access to pypi.org, files.pythonhosted.org, wheels.vllm.ai, huggingface.co.
- Disk: ~25 GB under `$ISO_ROOT` (two venvs ~8 GB each, uv cache, model 1.5 GB), ~20 GB under `$STAGE`.

## 1. Settings

```
cp -r vllm30056 /data/vllm30056 && cd /data/vllm30056   # any location; this is ISO_ROOT
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
Must end with `SETUP_LOCK_OK 30056`, after printing for both roles that the freeze equals the authors', that all vllm
files hash as in the authors' build, and the import check. `vllm/v1/structured_output/__init__.py` and the reasoning
parsers under `vllm/reasoning/` (and `vllm/_version.py`) differ between parent and fix (`env/files_differing.txt`).

## 3. Model (~3-10 min)

```
bash setup/get_model.sh      # Qwen/Qwen3-0.6B @ c1899de2...; ends with GET_MODEL_OK
```

## 4. Stage to local disk (~2-10 min)

```
bash setup/stage.sh          # ends with STAGE_OK 30056
```

## 5. Calibration, then a scoring job of the human fix (one GPU; ~40 + ~18 min)

```
bash harness/run.sh cal_g0 cal          # 25 runs
bash harness/run.sh s1_g0 score         # 10 runs; candidate = fix
```
Each prints its summary at the end (also `results/<label>/verdict.json`).
Compile caches are kept per role, host and GPU under `$STAGE/cache/`; every candidate run starts from a fresh copy of
the parent's cache on that GPU, so both arms run the same tuned kernels (see the card, "The gate"). Run the parent of a
GPU first (every mode does). Stop cleanly between runs: `touch $ISO_ROOT/STOP`.
A second GPU can run another job at the same time: `GPU=1 CPUS=<cores near GPU1> PORT=36500 bash harness/run.sh s2_g1 score`.
Transfer loads (8 requests per rep; no structured output, a negative control): `bash harness/run.sh tr_g0 transfer`.
One of them only: `ONLY=b8 bash harness/run.sh trb8_g0 transfer`.
Verdict `rerun:exact`: counts, settings and the correctness pass all passed, but exactly one timed rep of one run
differs from its paired parent (a rare flip of this load, measured in the card). Run the same job once more under a new
label (e.g. `s1b_g0`); its verdict is final, a second `fail:exact` included.
To re-derive a verdict with other thresholds: edit `harness/gate.json`, then `python3 -I harness/check.py $ISO_ROOT <label>`.


**Compile caches.** Every candidate run starts from a copy of the parent's compile cache on that GPU, and calibration
runs the warm-ups before the passes it compares (since 2026-11-25, after acc6 found that on a fresh stage the copy
failed and C15's first, cold passes were compared). On the authors' GPUs this task's correctness pass came out the same
with every cache, including an empty one (fix vs parent 24/24 on GPU2 and GPU3); other GPUs were not tried, and whether
`harness/ref_correct.json` (recorded on GPU3) holds on them is unknown.

## 6. Broken and cheating variants (the checks must reject them; ~8 min each)

```
bash harness/mkvariant.sh short && bash harness/run.sh v_short quick var_short
```
Variants: `short`, `noadvance`, `noparser` (what each does: `variants/apply.py`). `quick` mode = warm-up, correctness
passes, one timed pair.

## 7. Expected results (card)

| Quantity | Card | Pass if |
|---|---|---|
| Scoring job of the fix | see `CARD_vllm30056.md` (REF and band are also in `harness/gate.json`) | verdict `ties_ref` |
| Calibration A/A | see card | within the card's A/A spread |
| Checks on the fix | counts, settings, exact timed outputs, correctness pass (bit-exact): all pass | no `fail:` |
| Broken variants | every one `fail:<checks>` as listed in the card | same checks fail |

## Files

- `harness/run.sh` (job lists, STOP and busy-GPU checks, GPU lock), `harness/job.sh` (guarded run: busy-GPU refusal,
  foreign-process abort, VRAM watchdog, CPU pinning, clean environment), `harness/check.py` (timing, checks, verdict),
  `harness/gate.json` (REF, band, thresholds), `harness/ref_correct.json` (the authors' reference correctness pass; a
  candidate's pass must equal this job's parent pass or this one), `harness/mkvariant.sh`, `variants/apply.py`.
- `bench/offline_c5y.py` (workload, correctness pass), `bench/run_bench.sh`; `loads/*.env` (official load `cal.env`,
  correctness `cor.env`, transfer loads `tr_*.env`, `smoke.env`).
- `env/`: `build.env` (commits, wheel URLs and hashes, model revision), `deps.lock` (hashed), the authors' freezes and
  vllm file hashes. `setup/`: build, download, stage.
