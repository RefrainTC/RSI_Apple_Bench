# Environment spec for #12380

Everything `setup/setup_venvs.sh` and `setup/get_model.sh` install, and where each fact comes from. The machine-readable
form is `env/pkg.conf`; the locks and the authors' records are in `env/`.

## Source code

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/12380 "[perf] fix perf regression from #12253", merged 2025-01-24T03:34:27Z (GitHub API `merged_at`) |
| Human fix (merge commit) | `6dd94dbe94c1820a1e224cba65efcf0befa97995` (vLLM 0.6.6.post2.dev365) |
| Parent (first parent of the fix) | `0e74d797ce8618fdb685126e0ff8576fb966e6ad` (vLLM 0.6.6.post2.dev364) |
| Changed by the PR | `vllm/worker/model_runner.py` (4 lines); between the two wheels also `vllm/_C.abi3.so` (separate builds) and `vllm/_version.py` |

## Wheels (vLLM's per-commit builds)

| Role | URL | sha256 |
|---|---|---|
| parent | https://wheels.vllm.ai/0e74d797ce8618fdb685126e0ff8576fb966e6ad/vllm-0.6.6.post2.dev364%2Bg0e74d797-cp38-abi3-manylinux1_x86_64.whl | `f9f542a35315939fa30fcb47f18860c34d0360550c328b9f4a5ce6222d4addaf` |
| fix | https://wheels.vllm.ai/6dd94dbe94c1820a1e224cba65efcf0befa97995/vllm-0.6.6.post2.dev365%2Bg6dd94dbe-cp38-abi3-manylinux1_x86_64.whl | `9c99534fb83d58e8ee7fe928c1cc5229d58f1df9fd3bd580beeefe612815f021` |

The authors' download records are `env/wheel_12380_{parent,fix}.SOURCE.txt`. If wheels.vllm.ai ever drops these files,
`screen/fetch.py <pr> <role> <commit> <outdir>` shows how they were found (index page, then the nightly alias); any
replacement must have the same sha256.

## Python packages

- Interpreter: CPython **3.10.20** as installed by **uv 0.11.14** (`uv python install 3.10.20`, python-build-standalone).
- `env/12380_parent.lock.txt`, `env/12380_fix.lock.txt`: every package of the authors' venvs pinned to the authors'
  version, with the sha256 of every distribution file PyPI served for that version as of the cutoff. Installed with
  `uv pip install --require-hashes --no-deps --exclude-newer 2025-01-26` (the cutoff is the merge date + 2 days, as in
  the authors' build). The locks were generated from the authors' `uv pip freeze` output (`env/12380_*_freeze.txt`) with
  `uv pip compile --no-deps --generate-hashes --python-version 3.10 --python-platform x86_64-manylinux_2_28 --exclude-newer 2025-01-26`.
- Main versions: torch 2.5.1 (PyPI build, CUDA 12.4), transformers 4.48.1, triton 3.1.0, xformers 0.0.28.post3,
  xgrammar 0.1.11, numpy 1.26.4, huggingface-hub 0.27.1, tokenizers 0.21.0, setuptools 75.8.0. The fix venv has
  exactly the parent's packages apart from vllm (`env/12380_freeze.diff`).
- Overrides: `pyairports` is not installed (its old releases were deleted from PyPI) and is replaced by the authors'
  two-file stub (`pyairports/__init__.py`, `pyairports/airports.py` with `AIRPORT_LIST = []`); nothing in this task
  uses it. No other override is needed for this PR.
- Checks after the build: `uv pip freeze` must equal the authors' freeze (vllm line aside), and every file of the
  installed `vllm` package must hash as in `env/12380_{parent,fix}_vllm.sha256` (the authors' record of the venvs that
  produced the card). `setup_venvs.sh` fails otherwise.

How the authors built them originally (kept for the record, not needed): `screen/setup_prs.sh
"12380:6dd94dbe94c1820a1e224cba65efcf0befa97995:2025-01-24:0e74d797ce8618fdb685126e0ff8576fb966e6ad"`, which resolves
the dependencies with `--exclude-newer` = merge date + 2 days and constrains the fix venv to the parent's freeze. The
lock reproduces its result without re-resolving.

## Model

| | |
|---|---|
| Repo | `RedHatAI/Meta-Llama-3-8B-Instruct-FP8` (public; static FP8 activation scales) |
| Revision | `c5c6b5700a4178ef1fdae2ae37827382b90eb400` (the snapshot the authors used) |
| Files | 10 (config, generation config, tokenizer files, 2 safetensors shards, index, README, .gitattributes), 8.5 GB |
| Location | `$ISO_ROOT/hf/hub/models--RedHatAI--Meta-Llama-3-8B-Instruct-FP8`, staged to `$STAGE/hf/hub/`; jobs run with `HF_HUB_OFFLINE=1` |

## Runtime settings that matter for the numbers

- One GPU per job (`CUDA_VISIBLE_DEVICES=$GPU`), `gpu_memory_utilization` 0.85 (in `tasks/iso12380/runner.py`), the
  harness's VRAM watchdog at 92 % of the card.
- CPU pinning with `taskset -c $CPUS`. The parent is CPU-launch-bound (no CUDA graphs), so CPU contention shows up
  directly in its time; the authors pinned each GPU to one CCD of a Threadripper PRO 5975WX.
- Venvs, interpreter, model and compile caches on local NVMe (`$STAGE`); every run uses a fresh copy of the venv.
