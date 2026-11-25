# Environment spec for #14930

Everything `setup/setup_venvs.sh` and `setup/get_model.sh` install, and where each fact comes from. The machine-readable
form is `env/pkg.conf`; the locks and the authors' records are in `env/`.

## Source code

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/14930 "[V1][Spec Decode] Optimize Rejection Sampler with Triton Kernels", merged 2025-03-18T21:31:54Z (GitHub API `merged_at`) |
| Human fix (merge commit) | `99abb8b650c66664cdc84d815b7f306f33bd9881` (vLLM 0.8.0rc3.dev22) |
| Parent (first parent of the fix) | `3a1e6481586ed7f079275b5d5072a6e246af691e` (vLLM 0.8.0rc3.dev21) |
| Changed by the PR | 7 non-test files (~1100 lines): `vllm/envs.py`, `vllm/v1/outputs.py`, `vllm/v1/sample/ops/utils.py`, `vllm/v1/sample/rejection_sampler.py`, `vllm/v1/spec_decode/metadata.py`, `vllm/v1/spec_decode/utils.py`, `vllm/v1/worker/gpu_model_runner.py`; between the two wheels also `vllm/_C.abi3.so` (separate builds) and `vllm/_version.py` (`env/14930_vllm_pkg_files.txt`) |

## Wheels (vLLM's per-commit builds)

| Role | URL | sha256 |
|---|---|---|
| parent | https://wheels.vllm.ai/3a1e6481586ed7f079275b5d5072a6e246af691e/vllm-0.8.0rc3.dev21%2Bg3a1e6481-cp38-abi3-manylinux1_x86_64.whl | `60bc4576749d0adeb48ca265141297eb32e8116a685a431008f288d30e8fc338` |
| fix | https://wheels.vllm.ai/99abb8b650c66664cdc84d815b7f306f33bd9881/vllm-0.8.0rc3.dev22%2Bg99abb8b6-cp38-abi3-manylinux1_x86_64.whl | `57af8c20233e515597461a1212707664ecd0f794814bf5eff9f39a9ac14d7e34` |

The authors' download records are `env/wheel_14930_{parent,fix}.SOURCE.txt`. If wheels.vllm.ai ever drops these files,
`screen/fetch.py <pr> <role> <commit> <outdir>` shows how they were found (index page, then the nightly alias); any
replacement must have the same sha256.

## Python packages

- Interpreter: CPython **3.10.20** as installed by **uv 0.11.14** (`uv python install 3.10.20`, python-build-standalone).
- `env/14930_parent.lock.txt`, `env/14930_fix.lock.txt`: every package of the authors' venvs pinned to the authors'
  version, with the sha256 of every distribution file PyPI served for that version as of the cutoff. Installed with
  `uv pip install --require-hashes --no-deps --exclude-newer 2025-03-20` (the cutoff is the merge date + 2 days, as in
  the authors' build). The locks were generated from the authors' `uv pip freeze` output (`env/14930_*_freeze.txt`) with
  `uv pip compile --no-deps --generate-hashes --python-version 3.10 --python-platform x86_64-manylinux_2_28 --exclude-newer 2025-03-20`.
- Main versions: torch 2.6.0 (PyPI build, CUDA 12.4), transformers 4.49.0, triton 3.2.0, xformers 0.0.29.post2,
  xgrammar 0.1.17, numpy 1.26.4, huggingface-hub 0.29.3, tokenizers 0.21.1, setuptools 77.0.3. The fix venv has
  exactly the parent's packages apart from vllm (`env/14930_freeze.diff`).
- Overrides: `pyairports` is not installed (its old releases were deleted from PyPI) and is replaced by the authors'
  two-file stub (`pyairports/__init__.py`, `pyairports/airports.py` with `AIRPORT_LIST = []`); nothing in this task
  uses it.
- **Override `xgrammar==0.1.17`**: vLLM 0.8.0rc3 pins xgrammar 0.1.16, which (like 0.1.15) was deleted from PyPI. The
  authors installed 0.1.17 (`env/uv_overrides_14930.txt`); it is exempt from the date cutoff
  (`--exclude-newer-package xgrammar=2026-11-24`). The task does not use guided decoding.
- Checks after the build: `uv pip freeze` must equal the authors' freeze (vllm line aside), and every file of the
  installed `vllm` package must hash as in `env/14930_{parent,fix}_vllm.sha256` (the authors' record of the venvs that
  produced the card). `setup_venvs.sh` fails otherwise.

How the authors built them originally (kept for the record, not needed): `screen/setup_prs.sh
"14930:99abb8b650c66664cdc84d815b7f306f33bd9881:2025-03-18:3a1e6481586ed7f079275b5d5072a6e246af691e:xgrammar==0.1.17"`, which resolves
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

- One GPU per job (`CUDA_VISIBLE_DEVICES=$GPU`), V1 engine, `gpu_memory_utilization` 0.80 (the runner caps it at
  0.85), torch.compile level 3 + CUDA graphs, n-gram speculation k = 3; the harness's VRAM watchdog at 92 % of the card.
- CPU pinning with `taskset -c $CPUS`; the authors pinned each GPU to one CCD of a Threadripper PRO 5975WX.
- Venvs, interpreter, model and compile caches on local NVMe (`$STAGE`); every run uses a fresh copy of the venv.
