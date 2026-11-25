# Environment spec for #16436

Everything `setup/setup_venvs.sh` and `setup/get_model.sh` install, and where each fact comes from. The machine-readable
form is `env/pkg.conf`; the locks and the authors' records are in `env/`.

## Source code

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/16436 "[Core] Speed up decode by remove synchronizing operation in sampler", merged 2025-04-21T18:18:23Z (GitHub API `merged_at`) |
| Human fix (merge commit) | `299ebb62b269ce167eb1c71b5e39a1dc1f65ce1c` (vLLM 0.8.5.dev118) |
| Parent (first parent of the fix) | `f728ab8e3578c22b42ed53e51b5e8ec35328d8b9` (vLLM 0.8.5.dev117) |
| Changed by the PR | `vllm/model_executor/layers/utils.py` (14 lines); between the two wheels also `vllm/_C.abi3.so` (separate builds) and `vllm/_version.py` (`env/16436_vllm_pkg.diff`) |

## Wheels (vLLM's per-commit builds)

| Role | URL | sha256 |
|---|---|---|
| parent | https://wheels.vllm.ai/f728ab8e3578c22b42ed53e51b5e8ec35328d8b9/vllm-0.8.5.dev117%2Bgf728ab8e3-cp38-abi3-manylinux1_x86_64.whl | `6e5c56bacf294392a31a458f64ae23d9a072f522cf07a0e867f21bf65910b98f` |
| fix | https://wheels.vllm.ai/299ebb62b269ce167eb1c71b5e39a1dc1f65ce1c/vllm-0.8.5.dev118%2Bg299ebb62b-cp38-abi3-manylinux1_x86_64.whl | `3b92da226ad1e4b2ed431a4c33aa25bada22d0e68f7f93f2c0dc8ed3ba9f3dde` |

The authors' download records are `env/wheel_16436_{parent,fix}.SOURCE.txt`. If wheels.vllm.ai ever drops these files,
`screen/fetch.py <pr> <role> <commit> <outdir>` shows how they were found (index page, then the nightly alias); any
replacement must have the same sha256.

## Python packages

- Interpreter: CPython **3.10.20** as installed by **uv 0.11.14** (`uv python install 3.10.20`, python-build-standalone).
- `env/16436_parent.lock.txt`, `env/16436_fix.lock.txt`: every package of the authors' venvs pinned to the authors'
  version, with the sha256 of every distribution file PyPI served for that version as of the cutoff. Installed with
  `uv pip install --require-hashes --no-deps --exclude-newer 2025-04-23` (the cutoff is the merge date + 2 days, as in
  the authors' build). The locks were generated from the authors' `uv pip freeze` output (`env/16436_*_freeze.txt`) with
  `uv pip compile --no-deps --generate-hashes --python-version 3.10 --python-platform x86_64-manylinux_2_28 --exclude-newer 2025-04-23`.
- Main versions: torch 2.6.0 (PyPI build, CUDA 12.4), transformers 4.51.3, triton 3.2.0, xformers 0.0.29.post2,
  xgrammar 0.1.18, outlines 0.1.11, numpy 2.2.5, huggingface-hub 0.30.2, tokenizers 0.21.1, setuptools 79.0.1. The fix venv has
  exactly the parent's packages apart from vllm (`env/16436_freeze.diff`).
- Overrides: none (these venvs resolved without pyairports; `PYAIRPORTS_STUB=none`).
- Checks after the build: `uv pip freeze` must equal the authors' freeze (vllm line aside), and every file of the
  installed `vllm` package must hash as in `env/16436_{parent,fix}_vllm.sha256` (the authors' record of the venvs that
  produced the card). `setup_venvs.sh` fails otherwise.
- **Bytecode.** `env/pkg.conf` (`post_setup`) pre-compiles every module's `.pyc` (`python -m compileall`). The authors'
  venvs already held them from earlier use; without them the `vllm serve` processes write `.pyc` files into the run
  copy of the venv, the venv guard reports `changed_by_run` and every scored pair is flagged `venv_changed`
  (first package test, 2026-11-24). Bytecode does not change the code that runs (the `vllm` file check excludes `.pyc`).

How the authors built them originally (kept for the record, not needed): `t5scripts/setup.sh` then
`t5scripts/setup_fix.sh` (their T5 tooling; paths parameterized like the rest), which resolve the dependencies with
`--exclude-newer 2025-04-23`, constrain the fix venv to the parent's freeze and add setuptools. The
lock reproduces its result without re-resolving.

## Model

| | |
|---|---|
| Repo | `Qwen/Qwen2.5-1.5B-Instruct` (public) |
| Revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (the snapshot the authors used) |
| Files | 10 (config, generation config, tokenizer files, merges, vocab, one safetensors file, LICENSE, README, .gitattributes), 3.1 GB |
| Location | `$ISO_ROOT/hf/hub/models--Qwen--Qwen2.5-1.5B-Instruct`, staged to `$STAGE/hf/hub/`; jobs run with `HF_HUB_OFFLINE=1` |

## Runtime settings that matter for the numbers

- One GPU per job (`CUDA_VISIBLE_DEVICES=$GPU`). Server: `vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching
  --dtype float16 -O3 --gpu-memory-utilization 0.85` (the PR's command minus the Hopper-only FlashAttention-3 switch),
  with the model's generation config (repetition_penalty 1.1, the code path the PR changes). Client: `vllm bench serve`
  from the pristine venv, 100 random prompts, 1000 in / 100 out, 1 request/s, `ignore_eos`; metric = median TPOT.
- CPU pinning with `taskset -c $CPUS`; the authors pinned each GPU to one CCD of a Threadripper PRO 5975WX.
- Venvs, interpreter, model and compile caches on local NVMe (`$STAGE`); every run uses a fresh copy of the venv.
