# Run the #16436 task from this directory

- **PR:** https://github.com/vllm-project/vllm/pull/16436 ("[Core] Speed up decode by remove synchronizing operation in
  sampler"), merged **2025-04-21** (18:18 UTC). Human fix = merge commit `299ebb62b269ce167eb1c71b5e39a1dc1f65ce1c`;
  parent = `f728ab8e3578c22b42ed53e51b5e8ec35328d8b9` (vLLM 0.8.5.dev117).
- **What you get:** the scorer (the authors' shared harness `harness/`, version v4e), the job runner and the task
  definition that produced card 2 in `SAMPLE_CARDS.md`, with the authors' machine paths replaced by the variables in
  `env.sh` (file-by-file record in `provenance/`), and an exact environment spec (`ENV.md`, `env/`). The load is
  `vllm serve` (candidate code) measured by vLLM's own `bench serve` client from the pristine venv.
- **Expected result** (card): human-fix speedup (client median TPOT, parent / candidate) **1.0855 ± 1.0 %**
  (1.0746-1.0964) per scoring job; calibration REF 1.093 (1.090-1.094); calibration A/A max deviation 0.43 %.

## 0. What you need

| | |
|---|---|
| GPU | 1x NVIDIA RTX 4090 24 GB (numbers are for the 4090). Driver for CUDA 12.4 (>= 550). |
| CPU / RAM | x86_64 Linux, glibc >= 2.28. Pin the job to 8 cores + SMT siblings near the GPU (`CPUS`). ~32 GB RAM. |
| Tools | `bash`, GNU coreutils/findutils, `tar`, `flock`, `taskset`, `nvidia-smi`, `git`, `curl`; `python3` >= 3.9 on `PATH` (stdlib only; if missing, `env.sh` falls back to the CPython of step 2); **uv 0.11.14**. No CUDA toolkit, no sudo. |
| Network | PyPI, github.com (uv's CPython build), wheels.vllm.ai, huggingface.co (public model, no token). |
| Disk | `ISO_ROOT`: ~25 GB (venvs 2x ~6.5 GB, uv cache ~8 GB, model 3.1 GB, results ~0.3 GB per job). `STAGE` (fast local disk): ~30 GB. |

Install uv if you do not have it: `curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`.

## 1. Place the package and set the machine variables (1 min)

```bash
cp -r iso16436 /data/iso16436          # any persistent directory; this directory becomes ISO_ROOT
cd /data/iso16436
cp site.env.example site.env           # then edit: STAGE, GPU, CPUS, PORT (and UV if uv is not on PATH)
source env.sh                          # every later command assumes this shell
```

Variables (defaults in `env.sh`): `ISO_ROOT` = this directory (also the root of the HF cache `hf/`);
`STAGE` = `/tmp/iso16436` (exported as `P2T`); `GPU` = 0 (the authors calibrated on GPU0); `CPUS` = all CPUs (authors:
one CCD per GPU of a Threadripper PRO 5975WX, GPU0 `0-7,32-39`, GPU1 `8-15,40-47`); `PORT` = 47000 + 200 x GPU
(`VLLM_PORT`); the server itself listens on 18430 + GPU index (`tasks/iso16436/run.sh`); `UV` = `uv` on `PATH`. Run all commands from `$ISO_ROOT`.

## 2. Build the two venvs (CPU and network only; ~8 min)

```bash
bash setup/setup_venvs.sh              # log: setup/setup_venvs.log
```

CPython 3.10.20 via uv, then `venvs/16436_parent` and `venvs/16436_fix` from the hashed locks plus the per-commit vLLM
wheels, then pre-compiles the bytecode (see `ENV.md`, "Bytecode"). Must end with `rc=0` and print, for
both roles, "freeze: identical to the authors'" and "vllm files: ... hashes identical".

## 3. Download the model (~1 min, 3.1 GB)

```bash
bash setup/get_model.sh                # Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306; must print LFS blobs ok=yes
```

## 4. Stage to the fast disk (~4 min)

```bash
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/16436_parent venv:$ISO_ROOT/venvs/16436_fix hf:Qwen/Qwen2.5-1.5B-Instruct
```

## 5. Build the agent workspace and its pristine copy (<1 min)

```bash
bash harness/build_workspace.sh tasks/iso16436
```

The workspace is `tasks/iso16436/workspace`: `repo/vllm/` (the parent's Python source), `WORKLOAD.md`, `dev_run.sh` and
`instruction.md`. Its `dev_run.sh` measures the directory it sits in, so an agent may work in a copy of the workspace
(before 2026-11-25 it always measured `tasks/iso16436/workspace`). `instruction.md` is the public task's `instruction.md`
when this package sits as `package/` in a public task directory (next to that task's `task.toml`); otherwise it is
`tasks/iso16436/instruction.md`, the package's own task text. The script prints which one it used.

## 6. Calibrate: REF, A/A, references, manifests (one GPU, ~75 min)

```bash
bash harness/calibrate.sh tasks/iso16436 5 5          # log: tasks/iso16436/calib/calibrate.log
bash scripts/post_calib.sh iso16436                    # record, manifests (prints REF ... thresholds ...)
python3 -I -c "import json; c=json.load(open('tasks/iso16436/calib/calib.json')); print('REF', round(c['ref_speedup'],4), 'fix pairs', c['fix_pairs']['values'], 'A/A', c['aa']['values'], 'A/A max dev', round(c['noise_band'],4))"
```

Each run starts a server (~50 s) and sends 100 requests at 1 request/s with 1000-token prompts and 100 output tokens
(~165 s per run). Card: REF 1.093 (1.090-1.094), A/A max deviation 0.43 %. The single-calibration REF reads higher than
the 7-job median 1.0855 (parent TPOT drifts between jobs by ~0.3 %); use the multi-job value for verdicts.

`post_calib.sh` is steps 1-3 of the authors' `scripts/after_cal.sh` (re-derived calibration record, manifests); this
is an online task, so there are no output thresholds to calibrate.

## 7. Score the human fix (one GPU, ~19 min)

```bash
bash harness/mk_fix_ws.sh tasks/iso16436 $ISO_ROOT/fixws        # pristine workspace + the human fix
bash harness/score.sh tasks/iso16436 fix1 $ISO_ROOT/fixws
cat tasks/iso16436/results/fix1/summary.md
```

Gate, integrity check, correctness client run, 2 warm-ups, 2 ABBA pairs. Expected: `speedup` in **1.0746-1.0964**;
correctness pass, every pair `outcheck True` (≥ 90 % identical texts, TTFT guard ≤ 1.05x), integrity pre/post `True`.
The verdict compares with your own calibration REF and the default ±0.5 % band, so a job at the card's median reads
`above_baseline_below_ref` against a ~1.093 calibration; that is expected (the card's band is ±1.0 % around 1.0855).

## 8. Optional

- **A/A under scoring conditions:** `bash harness/score.sh tasks/iso16436 pw1 tasks/iso16436/workspace` (expected ~1.0, `no_gain`; card: 1.0035).
- **Transfer loads** (4 requests/s 1.08x, unlimited rate 1.12x, 400 output tokens 1.08x): add `--transfer` as the 4th
  argument of `score.sh`.
- **Several jobs:** more `score.sh` runs with new labels, then `python3 -I scripts/jobvar.py tasks/iso16436 fix1 fix2 ...`.
  Another GPU: prefix the command with all three variables, e.g. `GPU=1 CPUS=8-15,40-47 PORT=47200 bash ...`
  (variables set by the caller win over `site.env`), and keep one job per GPU.
- **Canaries** (card table): `bash scripts/make_canaries16436.sh <out dir>` makes the workspaces; `scripts/canaries16436.sh` is the authors' queue that scored them.
- **Not needed outside the authors' setup:** `harness/vault.sh`, `remote_*_entry.sh`, `remote_score_finish.sh` (hide
  references and the fix from an agent under trial; the authors' list of paths to seal is not shipped).

## Known behaviour

- `stage_tmp.sh` pre-creates `hf/hub/version.txt`, so the first job on a new machine is not voided by an audit event.
- `score.sh` ends with `exec score_phase2.sh`, so the job's compile-cache root (`$STAGE/cache/iso16436/job-<label>-*`) is
  left behind; delete old ones by hand.
- The effect is ~9 %, so 1 % of drift moves the reward by ~11 % of its range: judge a candidate against the multi-job
  REF, not one calibration. On the authors' second pod (n1) three later jobs read 1.0833-1.0934.
- This harness version (the shared v4e) has no STOP file: stop between jobs, not inside one.
