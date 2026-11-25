# Run the #13365 task from this directory

- **PR:** https://github.com/vllm-project/vllm/pull/13365 ("[V1][Spec Decode] Optimize N-gram matching with Numba"),
  merged **2025-02-18** (21:19 UTC). Human fix = merge commit `4c822298981a8f7521492075ff72659985fc4c3f`; parent =
  `c8d70e2437feecdb3762ce17298df33439ae1bd1` (vLLM 0.7.3.dev215, V1 engine, n-gram speculative decoding).
- **What you get:** the scorer, harness and operator scripts that produced the numbers on `CARD_iso13365.md`, with the
  authors' machine paths replaced by the variables in `env.sh` (file-by-file record in `provenance/`), the task
  definition (`tasks/iso13365/`), and an exact environment spec (`ENV.md`, `env/`).
- **Expected result** (card): human-fix speedup **1.0934 ± 0.5 %** (1.0879-1.0989) per scoring job; calibration REF
  1.0952 (1.0905-1.0975); calibration A/A within ~0.5 % (one authors' pair read 1.698 during a host stall, see below).

## 0. What you need

| | |
|---|---|
| GPU | 1x NVIDIA RTX 4090 24 GB (numbers are for the 4090). Driver for CUDA 12.4 (>= 550). Peak use 20.0 GB. |
| CPU / RAM | x86_64 Linux, glibc >= 2.28. The proposer runs on the CPU between GPU steps, so CPU contention inflates the REF: pin the job to 8 cores + SMT siblings near the GPU (`CPUS`) and keep the machine quiet. ~32 GB RAM. |
| Tools | `bash`, GNU coreutils/findutils, `tar`, `flock`, `taskset`, `nvidia-smi`, `git`, `curl`; `python3` >= 3.9 on `PATH` (stdlib only; if missing, `env.sh` falls back to the CPython of step 2); **uv 0.11.14**. No CUDA toolkit, no sudo. |
| Network | PyPI, github.com (uv's CPython build), wheels.vllm.ai, huggingface.co (public model, no token). |
| Disk | `ISO_ROOT`: ~30 GB (three venvs ~6.4 GB each, uv cache ~8 GB, model 3.1 GB, results ~0.2 GB per job). `STAGE` (fast local disk): ~30 GB. |

Install uv if you do not have it: `curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`.

## 1. Place the package and set the machine variables (1 min)

```bash
cp -r iso13365 /data/iso13365          # any persistent directory; this directory becomes ISO_ROOT
cd /data/iso13365
cp site.env.example site.env           # then edit: STAGE, GPU, CPUS, PORT (and UV if uv is not on PATH)
source env.sh                          # every later command assumes this shell
```

Variables (defaults in `env.sh`): `ISO_ROOT` = this directory; `STAGE` = `/tmp/iso13365` (exported as `P2T`); `GPU` = 0
(the authors used GPU1); `CPUS` = all CPUs (authors: main GPU1 `8-15,40-47`, n1 GPU1 `24-31,56-63`, one CCD of a
Threadripper PRO 5975WX); `PORT` = 47000 + 200 x GPU (`VLLM_PORT`); `UV` = `uv` on `PATH`. Run all commands from
`$ISO_ROOT`.

## 2. Build the venvs (CPU and network only; ~8 min, ~27 GB incl. the uv cache)

```bash
bash setup/setup_venvs.sh              # log: setup/setup_venvs.log
```

Builds `venvs/13365_parent` and `venvs/13365_fix` from the hashed locks plus the per-commit vLLM wheels, then the task's
parent venv **`venvs/13365_base` = parent + numba 0.60.0 / llvmlite 0.43.0 copied from the fix venv** (the human fix
imports numba, so both arms need it; the authors' `ops_iso13365/setup_main.sh` step 1). Must end with `rc=0` and print
"freeze: identical" and "vllm files: ... hashes identical" for both roles, and "base freeze: parent + numba/llvmlite".

## 3. Download the model (~1 min, 3.1 GB)

```bash
bash setup/get_model.sh                # Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306; must print LFS blobs ok=yes
```

## 4. Stage to the fast disk (~4 min)

```bash
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/13365_base venv:$ISO_ROOT/venvs/13365_fix hf:Qwen/Qwen2.5-1.5B-Instruct
```

The task runs `13365_base` (both arms; the fix is an overlay) and needs `13365_fix` for the oracle stage and the
venv-equivalence pair. Re-run after a reboot if `$STAGE` is wiped.

## 5. Build the agent workspace and its pristine copy (<1 min)

```bash
bash harness_iso13365/build_workspace.sh tasks/iso13365
```

## 6. Calibrate: REF, A/A, output references, manifests (one GPU, ~33 min)

```bash
bash ops_iso13365/calib.sh $GPU                     # log: tasks/iso13365/calib/calibrate.log, ops_iso13365/logs/ops.log
python3 -I -c "import json; c=json.load(open('tasks/iso13365/calib/calib.json')); print('REF', round(c['ref_speedup'],4), 'fix pairs', c['fix_pairs']['values'], 'A/A', c['aa']['values'], 'A/A max dev', round(c['noise_band'],4))"
```

Card: REF 1.0952 (1.0905-1.0975); A/A 1.0033, 1.0010, 1.0008, 0.9997 and one 1.698 from a host stall (the acceptance
test got 0.9931-0.9992). `noise_band` is the largest A/A deviation, not a noise estimate.

## 7. Score the human fix (one GPU, ~10 min)

```bash
bash ops_iso13365/score_fix.sh fix1 $GPU            # makes tasks/iso13365/fixws (pristine workspace + the fix), then score.sh
cat tasks/iso13365/results/fix1/summary.md
```

The fix workspace goes through the full gate. Integrity check, correctness run (5 prompt groups, exact), 2 warm-ups, 3
ABBA pairs. Expected: `speedup` in **1.0879-1.0989**, verdict `ties_ref`, correctness pass, every pair `outcheck True`,
integrity pre/post `True`, no `environment_suspect_pairs` (the parent-time sentinel; a job with suspect pairs should
be rerun, not counted).

## 8. Optional

- **A/A under scoring conditions:** `bash ops_iso13365/aa.sh aa1 $GPU 3` (3 parent/parent pairs, ~7 min; card ≤ 0.47 %).
- **Transfer loads** (random prompts 1.577x, late match 1.265x, batch 64 1.098x): `bash ops_iso13365/transfer.sh t1 $GPU 2`.
  The `long:IN=3968` transfer in `task.env` crashes both arms (prompt + output > max_model_len); the card's 1.095x used IN 3840.
- **Several jobs:** more `score_fix.sh` runs with new labels, then `python3 -I ops_iso13365/jobs_summary.py tasks/iso13365 fix1 fix2 ...`.
  Another GPU: prefix the command with all three variables, e.g. `GPU=1 CPUS=8-15,40-47 PORT=47200 bash ...`
  (variables set by the caller win over `site.env`), and keep one job per GPU.
- **Canaries C1-C6:** `bash ops_iso13365/make_canaries.sh` / `make_canaries2.sh`, then `bash ops_iso13365/score_ws.sh <label> <workspace> $GPU`.
- **Not needed outside the authors' setup:** `harness_iso13365/vault.sh`, `remote_*_entry.sh`, `remote_score_finish.sh`
  (hide references and the fix from an agent under trial); `ops_iso13365/setup_main.sh` (replaced by steps 2-5);
  `stage_n1.sh`, `n1_*.sh`, `main_queue*.sh`, `n1_queue*.sh`, `v2_*.sh`, `smoke.sh`, `collect.sh`, `final_report.sh`
  (the authors' run scripts for their two pods, kept for the record).

## Known behaviour

- `stage_tmp.sh` pre-creates `hf/hub/version.txt`, so the first job on a new machine is not voided by an audit event.
- `score.sh` now takes its per-GPU lock (`$STAGE/locks/iso13365_gpu<N>.lock`); in the authors' copy the `flock` was
  commented out by accident.
- `score.sh` ends with `exec score_phase2.sh`, so each job leaves its compile-cache root
  (`$STAGE/cache/iso13365/job-<label>-*`, ~5 MB); delete old ones by hand.
- `vllm.__version__` prints `dev` (these wheels carry no `_version.py`), not 0.7.3.dev215.
- To stop cleanly between GPU runs: `touch $ISO_ROOT/STOP_iso13365`.
