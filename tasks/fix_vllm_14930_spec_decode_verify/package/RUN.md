# Run the #14930 task from this directory

- **PR:** https://github.com/vllm-project/vllm/pull/14930 ("[V1][Spec Decode] Optimize Rejection Sampler with Triton
  Kernels"), merged **2025-03-18** (21:31 UTC). Human fix = merge commit `99abb8b650c66664cdc84d815b7f306f33bd9881`;
  parent = `3a1e6481586ed7f079275b5d5072a6e246af691e` (vLLM 0.8.0rc3.dev21, V1 engine, n-gram speculative decoding).
- **What you get:** the scorer, harness and operator scripts that produced the numbers on `CARD_iso14930.md`, with the
  authors' machine paths replaced by the variables in `env.sh` (file-by-file record in `provenance/`), the task
  definition (`tasks/iso14930/`, including the curated prompt pool `pool14930.py`), and an exact environment spec
  (`ENV.md`, `env/`).
- **Expected result** (card): human-fix speedup **1.0909 ± 0.5 %** (1.0854-1.0964) per scoring job; calibration REF
  1.0909 (1.087-1.096); calibration A/A max deviation ~0.2-0.3 %; pristine workspace scored as a candidate ~0.9985.

## 0. What you need

| | |
|---|---|
| GPU | 1x NVIDIA RTX 4090 24 GB (numbers are for the 4090). Driver for CUDA 12.4 (>= 550). Peak use 20.3 GB. |
| CPU / RAM | x86_64 Linux, glibc >= 2.28. Pin the job to 8 cores + SMT siblings near the GPU (`CPUS`). ~32 GB RAM. |
| Tools | `bash`, GNU coreutils/findutils, `tar`, `flock`, `taskset`, `nvidia-smi`, `git`, `curl`; `python3` >= 3.9 on `PATH` (stdlib only; if missing, `env.sh` falls back to the CPython of step 2); **uv 0.11.14**. No CUDA toolkit (Triton JIT-compiles with its bundled tools), no sudo. |
| Network | PyPI, github.com (uv's CPython build), wheels.vllm.ai, huggingface.co (public model, no token). |
| Disk | `ISO_ROOT`: ~27 GB (venvs 2x 7.1 GB, uv cache ~8 GB, model 3.1 GB, results ~0.2 GB per job). `STAGE` (fast local disk): ~33 GB (staged venvs, model, per-run venv copies, compile caches). |

Install uv if you do not have it: `curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`.

## 1. Place the package and set the machine variables (1 min)

```bash
cp -r iso14930 /data/iso14930          # any persistent directory; this directory becomes ISO_ROOT
cd /data/iso14930
cp site.env.example site.env           # then edit: STAGE, GPU, CPUS, PORT (and UV if uv is not on PATH)
source env.sh                          # every later command assumes this shell
```

Variables (defaults in `env.sh`): `ISO_ROOT` = this directory; `STAGE` = `/tmp/iso14930` (exported as `P2T`); `GPU` = 0;
`CPUS` = all CPUs (authors: one CCD per GPU on a Threadripper PRO 5975WX, GPU2 `16-23,48-55`, GPU3 `24-31,56-63`);
`PORT` = 47000 + 200 x GPU (`VLLM_PORT`; one per concurrent job; a shared port once cost the authors a job);
`UV` = `uv` on `PATH`. Run all commands from `$ISO_ROOT`.

## 2. Build the two venvs (CPU and network only; ~8 min, ~21 GB incl. the uv cache)

```bash
bash setup/setup_venvs.sh              # log: setup/setup_venvs.log
```

CPython 3.10.20 via uv, then `venvs/14930_parent` and `venvs/14930_fix` from the hashed locks plus the per-commit vLLM
wheels. **Includes the `xgrammar==0.1.17` override** (vLLM pins 0.1.16, which PyPI no longer serves). Must end with
`rc=0` and print, for both roles, "freeze: identical to the authors'" and "vllm files: ... hashes identical".

## 3. Download the model (~1 min, 3.1 GB)

```bash
bash setup/get_model.sh                # Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306; must print LFS blobs ok=yes
```

## 4. Stage to the fast disk (~3 min)

```bash
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/14930_parent venv:$ISO_ROOT/venvs/14930_fix hf:Qwen/Qwen2.5-1.5B-Instruct
```

Re-run after a reboot if `$STAGE` is wiped (finished items are skipped).

## 5. Build the agent workspace and its pristine copy (<1 min)

```bash
bash harness_iso14930/build_workspace.sh tasks/iso14930
```

## 6. Calibrate: REF, A/A, output thresholds, manifests (one GPU, ~42 min)

```bash
bash ops_iso14930/calib.sh $GPU                     # log: tasks/iso14930/calib/calibrate.log
python3 -I -c "import json; c=json.load(open('tasks/iso14930/calib/calib.json')); print('REF', round(c['ref_speedup'],4), 'fix pairs', c['fix_pairs']['values'], 'A/A', c['aa']['values'], 'A/A max dev', round(c['noise_band'],4))"
```

`calib.sh` runs `harness_iso14930/calibrate.sh` (3 correctness runs, 2 output references, 2 warm-ups, 5 ABBA fix/parent
pairs, 5 A/A pairs, 1 venv-equivalence pair), then the #14930 output thresholds (`calib_outputs14930.py`) and the
manifests. Card: REF 1.0909 (1.087-1.096), A/A max deviation 0.19 %.

## 7. Score the human fix (one GPU, ~13 min)

```bash
bash ops_iso14930/score_job.sh $GPU fix1 tasks/iso14930/calib/oracle_stage
cat tasks/iso14930/results/fix1/summary.md
```

The human fix is scored from the oracle stage with the gate bypassed (`STAGE:` mode of `score.sh`): the PR's code is
human-written Triton, which the kernel gate rejects for agent candidates by design (card, validity limit 1). Integrity
check, correctness run, 2 warm-ups, 3 ABBA pairs. Expected: `speedup` in **1.0854-1.0964**, verdict `ties_ref`,
correctness pass, every pair `outcheck True`, integrity pre/post `True`.

## 8. Optional

- **A/A under scoring conditions:** the pristine workspace through the normal gate, expected ~0.9985, `no_gain`:
  `bash ops_iso14930/score_job.sh $GPU pw1 tasks/iso14930/workspace`
- **Transfer loads** (temperature 1.0 / k = 5 / batch 16 / no priming; card 1.189 / 1.072 / 1.088 / 1.043): add
  `--transfer` as the 4th argument of `score_job.sh` (+8 GPU-min).
- **Several jobs and the multi-job REF:** more `score_job.sh` runs with new labels (`GPU`, `CPUS`, `PORT` per GPU), then
  `python3 -I ops_iso14930/jobvar14930.py tasks/iso14930 fix1 fix2 ...` (its GPU column assumes the authors' CCD map).
  Another GPU: prefix the command with all three variables, e.g. `GPU=1 CPUS=8-15,40-47 PORT=47200 bash ...`
  (variables set by the caller win over `site.env`), and keep one job per GPU.
- **Canaries C1-C3** (cheats in plain PyTorch, through the normal gate): `python3 -I ops_iso14930/mk_canaries.py
  tasks/iso14930/workspace <out dir>`, then `score_job.sh` on each. **Torch-only rewrite V1**: `ops_iso14930/mk_alt.py`.
- **Pool re-screen** (parent only, ~3 min): `bash ops_iso14930/poolscreen.sh $GPU 704`; the card's rule kept 674 of 704.
- **Not needed outside the authors' setup:** `harness_iso14930/vault.sh`, `remote_*_entry.sh`, `remote_score_finish.sh`
  (hide references and the fix from an agent under trial), `ops_iso14930/setup.sh` (created the harness copy and the
  task dir on the authors' pod; steps 5-6 replace it), `queue.sh`, `pack.sh`, `pilot.sh`, `smoke.sh`, `aa_study.sh`,
  `c1_repeat.sh` (the authors' run scripts, kept for the record).

## Known behaviour

- `stage_tmp.sh` pre-creates `hf/hub/version.txt`, so the first job on a new machine is not voided by an audit event.
- `score.sh` ends with `exec score_phase2.sh`, so the job's compile-cache root (`$STAGE/cache/iso14930/job-<label>-*`)
  is left behind; delete old ones by hand.
- Single reps can be 38-47 % slow ("tail flips", card): the median of 7 reps absorbs them. A busy host (venv builds,
  large copies) pushed one acceptance job 0.01 % below the band; keep the machine quiet during timed jobs.
- To stop cleanly between GPU runs: `touch $ISO_ROOT/STOP_iso14930`.
