# Run the #12380 task from this directory

- **PR:** https://github.com/vllm-project/vllm/pull/12380 ("[perf] fix perf regression from #12253"), merged
  **2025-01-24** (03:34 UTC). Human fix = merge commit `6dd94dbe94c1820a1e224cba65efcf0befa97995`; parent =
  `0e74d797ce8618fdb685126e0ff8576fb966e6ad` (vLLM 0.6.6.post2.dev364, V0 engine).
- **What you get:** the scorer, harness and operator scripts that produced the numbers on `CARD_iso12380.md`, with the
  authors' machine paths replaced by the variables in `env.sh` (file-by-file record in `provenance/`), the task
  definition (`tasks/iso12380/`), and an exact environment spec (`ENV.md`, `env/`).
- **Expected result** (card): human-fix speedup **1.507 ± 1.1 %** (1.490-1.524) per scoring job; calibration REF
  1.513; calibration A/A max deviation up to ~2.3 % (the CPU-bound parent is the noisy side; one acceptance run saw 4.2 %);
  pristine workspace scored as a candidate ~0.998.

## 0. What you need

| | |
|---|---|
| GPU | 1x NVIDIA RTX 4090 24 GB (any 24 GB card with FP8 support should run; the numbers are for the 4090). Driver for CUDA 12.4 (>= 550). Peak use 20.1 GB. |
| CPU / RAM | x86_64 Linux, glibc >= 2.28. The load is CPU-launch-bound on the parent side: pin the job to 8 cores + SMT siblings near the GPU (`CPUS`) and keep other work off them. ~32 GB RAM. |
| Tools | `bash`, GNU coreutils/findutils, `tar`, `flock`, `taskset`, `nvidia-smi`, `git`, `curl`; `python3` >= 3.9 on `PATH` (stdlib only; if missing, `env.sh` falls back to the CPython that step 2 installs); **uv 0.11.14**. No CUDA toolkit, no sudo. |
| Network | PyPI, github.com (uv's CPython build), wheels.vllm.ai, huggingface.co (public model, no token). |
| Disk | `ISO_ROOT` (any disk): ~31 GB (venvs 2x 6.7 GB, uv cache ~8 GB, model 8.5 GB, results ~0.5 GB per job). `STAGE` (fast local disk, NVMe): ~35 GB (staged venvs 13.4 GB, model 8.5 GB, per-run venv copies 13.4 GB, compile caches). |

Install uv if you do not have it: `curl -LsSf https://astral.sh/uv/0.11.14/install.sh | sh` (puts `uv` in `~/.local/bin`).

## 1. Place the package and set the machine variables (1 min)

```bash
cp -r iso12380 /data/iso12380          # any persistent directory; this directory becomes ISO_ROOT
cd /data/iso12380
cp site.env.example site.env           # then edit: STAGE, GPU, CPUS, PORT (and UV if uv is not on PATH)
source env.sh                          # every later command assumes this shell (scripts also source env.sh themselves)
```

Variables (defaults in `env.sh`): `ISO_ROOT` = this directory; `STAGE` = `/tmp/iso12380` (exported to the scripts as
`P2T`); `GPU` = 0; `CPUS` = all CPUs (the authors used one CCD per GPU: GPU0 `0-7,32-39`, GPU1 `8-15,40-47`, GPU2
`16-23,48-55`, GPU3 `24-31,56-63` on a Threadripper PRO 5975WX); `PORT` = 47000 + 200 x GPU (vLLM's `VLLM_PORT`; use a
different one per concurrent job); `UV` = `uv` on `PATH`. Run all commands below from `$ISO_ROOT`.

## 2. Build the two venvs (CPU and network only; ~8 min, ~21 GB in `ISO_ROOT` incl. the uv cache)

```bash
bash setup/setup_venvs.sh              # log: setup/setup_venvs.log
```

Installs CPython 3.10.20 with uv, then `venvs/12380_parent` and `venvs/12380_fix` from the hashed locks
(`env/12380_*.lock.txt`, `--no-deps`, every wheel sha256-checked) plus the per-commit vLLM wheel (URL and sha256 in
`env/pkg.conf`). It must end with `rc=0` and print, for both roles, "freeze: identical to the authors'" and "vllm files:
... hashes identical to the authors' record". The two venvs differ only in `vllm/_C.abi3.so`, `vllm/_version.py` and
`vllm/worker/model_runner.py` (`env/12380_vllm_pkg_files.txt`).

## 3. Download the model (~3-5 min, 8.5 GB)

```bash
bash setup/get_model.sh                # RedHatAI/Meta-Llama-3-8B-Instruct-FP8 @ c5c6b5700a4178ef1fdae2ae37827382b90eb400
```

It must print `LFS blobs ok=yes`.

## 4. Stage venvs, interpreter and model to the fast disk (~5 min, ~22 GB in `STAGE`)

```bash
bash scripts/stage_tmp.sh py venv:$ISO_ROOT/venvs/12380_parent venv:$ISO_ROOT/venvs/12380_fix hf:RedHatAI/Meta-Llama-3-8B-Instruct-FP8
```

Copies are verified (file count + bytes) and the venvs are relocated to `$STAGE/uvpy`. `$STAGE` is wiped by a reboot
on most machines: re-run this step then (it skips finished items).

## 5. Build the agent workspace and its pristine copy (<1 min)

```bash
bash harness_iso12380/build_workspace.sh tasks/iso12380
```

## 6. Calibrate: REF, A/A, output thresholds, manifests (one GPU, ~30 min)

```bash
bash harness_iso12380/calibrate.sh tasks/iso12380 5 5      # log: tasks/iso12380/calib/calibrate.log
python3 -I -c "import json; c=json.load(open('tasks/iso12380/calib/calib.json')); print('REF', round(c['ref_speedup'],4), 'fix pairs', c['fix_pairs']['values'], 'A/A', c['aa']['values'], 'A/A max dev', round(c['noise_band'],4))"
```

25 GPU runs: 3 correctness runs, 2 output-reference runs, 2 warm-ups, 5 ABBA fix/parent pairs, 5 A/A pairs, 1
venv-equivalence pair. Card: REF 1.513 (1.496-1.524), A/A 0.977-1.013 (max deviation 2.3 %), outputs bit-identical.

## 7. Score the human fix as a candidate (one GPU, ~9 min)

```bash
bash harness_iso12380/mk_fix_ws.sh tasks/iso12380 $ISO_ROOT/fixws          # pristine workspace + the human fix
VERDICT_BAND=0.011 bash harness_iso12380/score.sh tasks/iso12380 fix1 $ISO_ROOT/fixws
cat tasks/iso12380/results/fix1/summary.md
```

Gate, integrity check, correctness run, 2 warm-ups, 3 ABBA pairs. Expected: `speedup` in **1.490-1.524**, verdict
`ties_ref` or `above_ref` (the verdict compares with your own calibration REF; `VERDICT_BAND=0.011` is the card's
±1.1 % band, the harness default is ±0.5 %), correctness pass, every pair `outcheck True`, integrity pre/post `True`.

## 8. Optional

- **A/A under scoring conditions:** score the pristine workspace, expected ~0.998, verdict `no_gain`:
  `bash harness_iso12380/score.sh tasks/iso12380 pw1 tasks/iso12380/workspace`
- **Transfer loads** (batch 1 / batch 32 / 512 output tokens, card 1.52x / 1.40x / 1.50x): add `--transfer` as the
  4th argument of `score.sh` (+8 min).
- **Several jobs:** run `score.sh` with new labels on other GPUs (`GPU=`, `CPUS=`, `PORT=` per job), then
  `python3 -I harness_iso12380/jobvar.py tasks/iso12380 fix1 fix2 ...` for the multi-job REF, SD and band.
  Another GPU: prefix the command with all three variables, e.g. `GPU=1 CPUS=8-15,40-47 PORT=47200 bash ...`
  (variables set by the caller win over `site.env`), and keep one job per GPU.
- **Canaries K1-K5** (cheats the scorer must catch, card table): `bash ops_iso12380/make_canaries.sh <out dir>` makes
  the workspaces; score each with `score.sh`.
- **Not needed outside the authors' setup:** `harness_iso12380/vault.sh`, `remote_*_entry.sh`, `remote_score_finish.sh`
  (they hide the references and the human fix from an agent under trial and move results to the authors' workstation),
  and the `ops_iso12380/` drivers (`driver_*.sh`, `stage_n1.sh`, `n1_pycfix.sh`, `smoke*.sh`, `extra_*.sh`), which are the
  authors' own run scripts for their two pods, kept for the record.

## Known behaviour

- The first job on a newly staged machine used to be voided (`audit_flag`) because transformers writes
  `hf/hub/version.txt`; `stage_tmp.sh` now pre-creates that file.
- `score.sh` ends with `exec score_phase2.sh`, so its cleanup trap does not delete the job's compile-cache root
  (`$STAGE/cache/iso12380/job-<label>-*`, < 1 MB here). Delete old ones by hand if you run many jobs.
- To stop cleanly between GPU runs: `touch $ISO_ROOT/STOP_iso12380` (remove it to continue).
- Do not build venvs or copy large files on the machine while a timed job runs: the parent side is CPU-bound and such
  load produced the only out-of-band A/A values in the acceptance tests.
