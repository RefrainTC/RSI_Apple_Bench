# Environment spec for #13365

Everything `setup/setup_venvs.sh` and `setup/get_model.sh` install, and where each fact comes from. The machine-readable
form is `env/pkg.conf`; the locks and the authors' records are in `env/`.

## Source code

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/13365 "[V1][Spec Decode] Optimize N-gram matching with Numba", merged 2025-02-18T21:19:58Z (GitHub API `merged_at`) |
| Human fix (merge commit) | `4c822298981a8f7521492075ff72659985fc4c3f` (vLLM 0.7.3.dev216) |
| Parent (first parent of the fix) | `c8d70e2437feecdb3762ce17298df33439ae1bd1` (vLLM 0.7.3.dev215) |
| Changed by the PR | `vllm/v1/spec_decode/ngram_proposer.py`, `vllm/v1/worker/gpu_model_runner.py` and the `numba==0.60.0` requirement (3 files, 127 lines); between the two wheels also `vllm/_C.abi3.so` and `vllm/_moe_C.abi3.so` (separate builds; the task runs the parent venv plus a Python overlay, so they do not matter) |

## Wheels (vLLM's per-commit builds)

| Role | URL | sha256 |
|---|---|---|
| parent | https://wheels.vllm.ai/c8d70e2437feecdb3762ce17298df33439ae1bd1/vllm-0.7.3.dev215%2Bgc8d70e24-cp38-abi3-manylinux1_x86_64.whl | `48513f1cbff0aa090b1df516117af66fd6c529b20efbb58ab82943aa6c7cab91` |
| fix | https://wheels.vllm.ai/4c822298981a8f7521492075ff72659985fc4c3f/vllm-0.7.3.dev216%2Bg4c822298-cp38-abi3-manylinux1_x86_64.whl | `45d18518b360753aec1dea57c15ea8d5a024c4b7ba0d0852dd1aa66f284aadb3` |

The authors' download records are `env/wheel_13365_{parent,fix}.SOURCE.txt`. If wheels.vllm.ai ever drops these files,
`screen/fetch.py <pr> <role> <commit> <outdir>` shows how they were found (index page, then the nightly alias); any
replacement must have the same sha256.

## Python packages

- Interpreter: CPython **3.10.20** as installed by **uv 0.11.14** (`uv python install 3.10.20`, python-build-standalone).
- `env/13365_parent.lock.txt`, `env/13365_fix.lock.txt`: every package of the authors' venvs pinned to the authors'
  version, with the sha256 of every distribution file PyPI served for that version as of the cutoff. Installed with
  `uv pip install --require-hashes --no-deps --exclude-newer 2025-02-20` (the cutoff is the merge date + 2 days, as in
  the authors' build). The locks were generated from the authors' `uv pip freeze` output (`env/13365_*_freeze.txt`) with
  `uv pip compile --no-deps --generate-hashes --python-version 3.10 --python-platform x86_64-manylinux_2_28 --exclude-newer 2025-02-20`.
- Main versions: torch 2.5.1 (PyPI build, CUDA 12.4), transformers 4.49.0, triton 3.1.0, xformers 0.0.28.post3,
  xgrammar 0.1.11, numpy 1.26.4, huggingface-hub 0.29.1, tokenizers 0.21.0, setuptools 75.8.0. The fix venv has the
  parent's packages plus numba 0.60.0 and llvmlite 0.43.0 (`env/13365_freeze.diff`).
- **Task venv `13365_base`** = `13365_parent` + the `numba`, `numba-0.60.0.dist-info`, `llvmlite`,
  `llvmlite-0.43.0.dist-info`, `llvmlite.libs` directories copied from `13365_fix` (`post_setup` in `env/pkg.conf`, the
  authors' `ops_iso13365/setup_main.sh` step 1). Both arms of every timed pair run this venv; the fix is a Python
  overlay. Its freeze must equal the fix freeze without vllm.
- Overrides: `pyairports` is not installed (its old releases were deleted from PyPI) and is replaced by the authors'
  two-file stub (`pyairports/__init__.py`, `pyairports/airports.py` with `AIRPORT_LIST = []`); nothing in this task
  uses it. No other override is needed for this PR.
- Checks after the build: `uv pip freeze` must equal the authors' freeze (vllm line aside), and every file of the
  installed `vllm` package must hash as in `env/13365_{parent,fix}_vllm.sha256` (the authors' record of the venvs that
  produced the card). `setup_venvs.sh` fails otherwise.

How the authors built them originally (kept for the record, not needed): `screen/setup_prs.sh
"13365:4c822298981a8f7521492075ff72659985fc4c3f:2025-02-18:c8d70e2437feecdb3762ce17298df33439ae1bd1"`, which resolves
the dependencies with `--exclude-newer` = merge date + 2 days and constrains the fix venv to the parent's freeze. The
lock reproduces its result without re-resolving.

## Model

| | |
|---|---|
| Repo | `Qwen/Qwen2.5-1.5B-Instruct` (public) |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (the snapshot the authors used) |
| Files | 10 (config, generation config, tokenizer files, merges, vocab, one safetensors file, LICENSE, README, .gitattributes), 3.1 GB |
| Location | `$ISO_ROOT/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct`, staged to `$STAGE/hf/hub/`; jobs run with `HF_HUB_OFFLINE=1` |

## Runtime settings that matter for the numbers

- One GPU per job (`CUDA_VISIBLE_DEVICES=$GPU`), V1 engine with the engine core in-process
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, set by the task), n-gram speculation k = 5, `gpu_memory_utilization` 0.80,
  torch.compile + CUDA graphs; the harness's VRAM watchdog at 92 % of the card.
- CPU pinning with `taskset -c $CPUS`. The proposer runs on the CPU between GPU steps and the parent does more CPU work
  than the fix, so CPU contention raises the measured speedup (card: 1.117 during a load spike).
- Venvs, interpreter, model and compile caches on local NVMe (`$STAGE`); every run uses a fresh copy of the venv.
