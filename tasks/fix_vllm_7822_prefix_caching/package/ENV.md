# Environment spec for #7822

Everything `setup/setup_venvs.sh` and `setup/get_model.sh` install, and where each fact comes from. The machine-readable
form is `env/pkg.conf`; the locks and the authors' records are in `env/`.

## Source code

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/7822 "[Performance][BlockManagerV2] Mark prefix cache block as computed after schedule", merged 2024-08-26T18:24:53Z (GitHub API `merged_at`) |
| Human fix (merge commit) | `2deb029d115dadd012ce5ea70487a207cb025493` (wheel version 0.5.5) |
| Parent (first parent of the fix) | `029c71de11bc3bcf84a1b3cf9d91e79ab6949799` (wheel version 0.5.5) |
| Changed by the PR | `vllm/core/block/prefix_caching_block.py`, `vllm/core/block_manager_v2.py` (33 lines); between the two wheels also `vllm/commit_id.py` (`env/7822_vllm_pkg.diff`) |

## Wheels (vLLM's per-commit builds)

| Role | URL | sha256 |
|---|---|---|
| parent | https://wheels.vllm.ai/029c71de11bc3bcf84a1b3cf9d91e79ab6949799/vllm-0.5.5-cp38-abi3-manylinux1_x86_64.whl | `c7e6ff557caec8b283984db1257da410151c3725ca92827b7e82bad3656272f7` |
| fix | https://wheels.vllm.ai/2deb029d115dadd012ce5ea70487a207cb025493/vllm-0.5.5-cp38-abi3-manylinux1_x86_64.whl | `b61bbdf2405cce367eea673ebb0e171c894884fab740c20c7590605b93c82de0` |

The authors' download records are `env/wheel_7822_{parent,fix}.SOURCE.txt`. If wheels.vllm.ai ever drops these files,
`screen/fetch.py <pr> <role> <commit> <outdir>` shows how they were found (index page, then the nightly alias); any
replacement must have the same sha256.

## Python packages

- Interpreter: CPython **3.10.20** as installed by **uv 0.11.14** (`uv python install 3.10.20`, python-build-standalone).
- `env/7822_parent.lock.txt`, `env/7822_fix.lock.txt`: every package of the authors' venvs pinned to the authors'
  version, with the sha256 of every distribution file PyPI served for that version as of the cutoff. Installed with
  `uv pip install --require-hashes --no-deps --exclude-newer 2024-08-28` (the cutoff is the merge date + 2 days, as in
  the authors' build). The locks were generated from the authors' `uv pip freeze` output (`env/7822_*_freeze.txt`) with
  `uv pip compile --no-deps --generate-hashes --python-version 3.10 --python-platform x86_64-manylinux_2_28 --exclude-newer 2024-08-28`.
- Main versions: torch 2.4.0 (PyPI build, CUDA 12.1), transformers 4.44.2, triton 3.0.0, xformers 0.0.27.post2,
  vllm-flash-attn 2.6.1, outlines 0.0.46, numpy 1.26.4, huggingface-hub 0.24.6, tokenizers 0.19.1, setuptools 74.0.0. The fix venv has
  exactly the parent's packages apart from vllm (`env/7822_freeze.diff`).
- Overrides: `pyairports` is not installed (its old releases were deleted from PyPI) and is replaced by the authors'
  two-file stub (`pyairports/__init__.py`, `pyairports/airports.py` with `AIRPORT_LIST = []`; the R4090/T5 wording,
  `PYAIRPORTS_STUB=t5`); outlines 0.0.46 imports it, nothing in this task uses it. No other override is needed for this PR.
- Checks after the build: `uv pip freeze` must equal the authors' freeze (vllm line aside), and every file of the
  installed `vllm` package must hash as in `env/7822_{parent,fix}_vllm.sha256` (the authors' record of the venvs that
  produced the card). `setup_venvs.sh` fails otherwise.
- **`vllm-flash-attn` 2.6.1 must be present.** vLLM 0.5.5 takes FlashAttention-2 from this separate package; if it is
  missing, vLLM logs "Cannot use FlashAttention-2 backend because the vllm_flash_attn package is not found" and falls
  back to xformers plus a Triton prefix-prefill kernel. That changes the numbers (REF ~1.41 instead of 1.48) and the
  outputs (the scoring job then fails the output check), and the first Triton compile shows up as a `ptxas` exec in the
  audit. `env/pkg.conf` (`post_setup`) therefore checks that `vllm_flash_attn` imports and that its 3 files hash as in
  the authors' venvs (`env/7822_{parent,fix}_flashattn.sha256`, made from the authors' venvs on 2026-11-24).

How the authors built them originally (kept for the record, not needed): `t5scripts/setup.sh` then
`t5scripts/setup_fix.sh` (their T5 tooling; paths parameterized like the rest), which resolve the dependencies with
`--exclude-newer 2024-08-28`, constrain the fix venv to the parent's freeze and add setuptools. The
lock reproduces its result without re-resolving.

## Model

| | |
|---|---|
| Repo | `RedHatAI/Meta-Llama-3-8B-Instruct-FP8` (public; static FP8 activation scales) |
| Revision | `c5c6b5700a4178ef1fdae2ae37827382b90eb400` (the snapshot the authors used) |
| Files | 10 (config, generation config, tokenizer files, 2 safetensors shards, index, README, .gitattributes), 8.5 GB |
| Location | `$ISO_ROOT/hf/hub/models--RedHatAI--Meta-Llama-3-8B-Instruct-FP8`, staged to `$STAGE/hf/hub/`; jobs run with `HF_HUB_OFFLINE=1` |

## Benchmark script

`bench/7822_parent_benchmark_prefix_caching.py` is vLLM's `benchmarks/benchmark_prefix_caching.py` at the parent commit
(https://raw.githubusercontent.com/vllm-project/vllm/029c71de11bc3bcf84a1b3cf9d91e79ab6949799/benchmarks/benchmark_prefix_caching.py,
identical at the fix commit). The runner imports it to build the engine as upstream does (+ the PR's
`--use-v2-block-manager`). It is the authors' downloaded copy, unchanged.

## Runtime settings that matter for the numbers

- One GPU per job (`CUDA_VISIBLE_DEVICES=$GPU`), `enforce_eager=True`, prefix caching, v2 block manager,
  `gpu_memory_utilization` 0.85; the harness's VRAM watchdog at 92 % of the card.
- CPU pinning with `taskset -c $CPUS`; the authors pinned each GPU to one CCD of a Threadripper PRO 5975WX.
- Venvs, interpreter, model and compile caches on local NVMe (`$STAGE`); every run uses a fresh copy of the venv.
