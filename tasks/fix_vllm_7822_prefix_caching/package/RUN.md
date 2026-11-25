# Run the #7822 task from this directory

- **PR:** https://github.com/vllm-project/vllm/pull/7822 ("[Performance][BlockManagerV2] Mark prefix cache block as
  computed after schedule"), merged **2024-08-26** (18:24 UTC). Human fix = merge commit
  `2deb029d115dadd012ce5ea70487a207cb025493`; parent = `029c71de11bc3bcf84a1b3cf9d91e79ab6949799` (vLLM 0.5.5, V0).
- **What you get:** the scorer (the authors' shared harness `harness/`, version v4e), the job runner and the task
  definition that produced card 1 in `SAMPLE_CARDS.md`, with the authors' machine paths replaced by the variables in
  `env.sh` (file-by-file record in `provenance/`), the upstream `benchmarks/benchmark_prefix_caching.py` at the parent
  commit (`bench/`, the engine construction the runner reuses), and an exact environment spec (`ENV.md`, `env/`).
- **Expected result** (card): human-fix speedup **1.479 ± 0.5 %** (1.4716-1.4864) per scoring job; calibration REF
  1.480 (1.474-1.484); calibration A/A max deviation 0.16 %.

## 0. What you need

| | |
|---|---|
| GPU | 1x NVIDIA RTX 4090 24 GB (numbers are for the 4090). Driver for CUDA 12.1+ (>= 530). |
| CPU / RAM | x86_64 Linux, glibc >= 2.28. Pin the job to 8 cores + SMT siblings near the GPU (`CPUS`). ~32 GB RAM. |
| Tools | `bash`, GNU coreutils/findutils, `tar`, `flock`, `taskset`, `nvidia-smi`, `git`, `curl`; `python3` >= 3.9 on `PATH` (stdlib only; if missing, `env.sh` falls back to the CPython of step 2); **uv 0.11.14**. No CUDA toolkit, no sudo. |
| Network | PyPI, github.com (uv's CPython build), wheels.vllm.ai, huggingface.co (public model, no token). |
| Disk | `ISO_ROOT`: ~30 GB (venvs 2x ~5.5 GB, uv cache ~7 GB, model 8.5 GB, results ~0.3 GB per job). `STAGE` (fast local disk): ~32 GB. |

Install uv if you do not have it: `curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh`.

## 1. Place the package and set the machine variables (1 min)

```bash
cp -r iso7822 /data/iso7822            # any persistent directory; this directory becomes ISO_ROOT
cd /data/iso7822
cp site.env.example site.env           # then edit: STAGE, GPU, CPUS, PORT (and UV if uv is not on PATH)
source env.sh                          # every later command assumes this shell
```

Variables (defaults in `env.sh`): `ISO_ROOT` = this directory (also the root of the HF cache `hf/` and of `bench/`);
`STAGE` = `/tmp/iso7822` (exported as `P2T`); `GPU` = 0 (the authors calibrated on GPU1); `CPUS` = all CPUs (authors:
one CCD per GPU of a Threadripper PRO 5975WX, GPU0 `0-7,32-39`, GPU1 `8-15,40-47`); `PORT` = 47000 + 200 x GPU
(`VLLM_PORT`); `UV` = `uv` on `PATH`. Run all commands from `$ISO_ROOT`.

## 2. Build the two venvs (CPU and network only; ~6-15 min)

```bash
bash setup/setup_venvs.sh              # log: setup/setup_venvs.log
```

CPython 3.10.20 via uv, then `venvs/7822_parent` and `venvs/7822_fix` from the hashed locks plus the per-commit vLLM
wheels (both report version 0.5.5). `pyairports` is replaced by the authors' stub. Must end with `rc=0` and print, for
both roles, "freeze: identical to the authors'", "vllm files: ... hashes identical" and "vllm_flash_attn files: 3 hashes
identical" (without `vllm-flash-attn` vLLM 0.5.5 silently uses another attention path; see `ENV.md`).

## 3. Download the model (~3-5 min, 8.5 GB)

```bash
bash setup/get_model.sh                # RedHatAI/Meta-Llama-3-8B-Instruct-FP8 @ c5c6b5700a4178ef1fdae2ae37827382b90eb400; must print LFS blobs ok=yes
```

## 4. Stage to the fast disk (~4 min)

```bash
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/7822_parent venv:$ISO_ROOT/venvs/7822_fix hf:RedHatAI/Meta-Llama-3-8B-Instruct-FP8
```

## 5. Build the agent workspace and its pristine copy (<1 min)

```bash
bash harness/build_workspace.sh tasks/iso7822
```

## 6. Calibrate: REF, A/A, output thresholds, manifests (one GPU, ~15 min)

```bash
bash harness/calibrate.sh tasks/iso7822 5 5          # log: tasks/iso7822/calib/calibrate.log
bash scripts/post_calib.sh iso7822                    # record, output thresholds, manifests (prints REF ... thresholds ...)
python3 -I -c "import json; c=json.load(open('tasks/iso7822/calib/calib.json')); print('REF', round(c['ref_speedup'],4), 'fix pairs', c['fix_pairs']['values'], 'A/A', c['aa']['values'], 'A/A max dev', round(c['noise_band'],4))"
```

Card: REF 1.480 (1.474-1.484), A/A max deviation 0.16 %.

`post_calib.sh` is steps 1-3 of the authors' `scripts/after_cal.sh`. **Do not skip it:** it calibrates the scored-output
thresholds (`calib_outputs.py`; card: tie 0.8125 nats, min exact 0.1902). Without them `outcheck.py` uses strict
defaults and the human fix fails every pair as `output_mismatch` (72 of 100 fix outputs diverge at near-ties, card).

## 7. Score the human fix (one GPU, ~5 min)

```bash
bash harness/mk_fix_ws.sh tasks/iso7822 $ISO_ROOT/fixws        # pristine workspace + the human fix
bash harness/score.sh tasks/iso7822 fix1 $ISO_ROOT/fixws
cat tasks/iso7822/results/fix1/summary.md
```

Gate, integrity check, correctness run, 2 warm-ups, 3 ABBA pairs. Expected: `speedup` in **1.4716-1.4864**, verdict
`ties_ref` (or `above_ref` / `above_baseline_below_ref` by a hair: the verdict compares with your own calibration REF
and the default ±0.5 % band), correctness pass, every pair `outcheck True`, integrity pre/post `True`.

## 8. Optional

- **A/A under scoring conditions:** `bash harness/score.sh tasks/iso7822 pw1 tasks/iso7822/workspace` (expected ~1.0, `no_gain`).
- **Transfer loads** (10 output tokens 3.93x, 300 requests 1.55x, one identical prompt x100 1.49x): add `--transfer`
  as the 4th argument of `score.sh`.
- **Several jobs:** more `score.sh` runs with new labels, then `python3 -I scripts/jobvar.py tasks/iso7822 fix1 fix2 ...`.
  Another GPU: prefix the command with all three variables, e.g. `GPU=1 CPUS=8-15,40-47 PORT=47200 bash ...`
  (variables set by the caller win over `site.env`), and keep one job per GPU.
- **Canaries C1-C12** (card table): `bash harness/make_canaries.sh tasks/iso7822 <out dir>`, then `score.sh` each.
- **Not needed outside the authors' setup:** `harness/vault.sh`, `remote_*_entry.sh`, `remote_score_finish.sh` (hide
  references and the fix from an agent under trial; the authors' list of paths to seal is not shipped).

## Known behaviour

- `stage_tmp.sh` pre-creates `hf/hub/version.txt`, so the first job on a new machine is not voided by an audit event.
- `score.sh` ends with `exec score_phase2.sh`, so the job's compile-cache root (`$STAGE/cache/iso7822/job-<label>-*`) is
  left behind; delete old ones by hand.
- Host matters a little: on the authors' second pod (n1) four later jobs read 1.471-1.474, at or just under the band's
  lower edge. Calibrate and score on the same machine.
- This harness version (the shared v4e) has no STOP file: stop between jobs, not inside one.
- No run's `job_stdout.log` may say `Cannot use FlashAttention-2 backend` / `Using XFormers backend`: that means the
  venv lacks `vllm-flash-attn` (first package test, 2026-11-24: REF 1.41, scoring job failed the output check); rebuild the venvs.
