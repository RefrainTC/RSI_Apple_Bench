# Environment spec for #6994

Everything `b3/scripts/setup_lock.sh` and `get_model.sh` need is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/sgl-project/sglang/pull/6994 ("[Perf] Refactor LoRAManager to eliminate stream syncs and redundant computations"), merged 2025-06-11T23:18:57Z |
| Fix (merge commit) | `021f76e4f49861b2e9ea9ccff06a46d577e3c548` (SGLang 0.4.7) |
| Parent (first parent) | `777688b8929c877e4e28c2eac208d776abe4c3af` |
| Changed by the PR | `python/sglang/srt/lora/lora_manager.py`, `python/sglang/srt/lora/mem_pool.py` |

## SGLang sources

SGLang has no per-commit wheels. Each role's package is built from the commit's source tarball,
`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`, of which only `python/` is extracted and built by the
setuptools backend its `pyproject.toml` declares (nothing else from the tree is run). GitHub does not guarantee
byte-stable tarballs, so the check is per file: every extracted file must hash as in the authors' record
`env/6994_{parent,fix}_src.sha256` (and the file count must match).

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython, `UV_PYTHON_PREFERENCE=only-managed`).
- `env/6994_deps.lock`: 133 pinned packages with sha256 hashes, identical for parent and fix (the authors pinned the fix
  venv to the parent's freeze). Main pins: torch 2.7.1 (cu126 wheels from PyPI), torchvision 0.22.1, torchaudio 2.7.1,
  triton 3.3.1, sgl-kernel 0.1.7, torchao 0.9.0, xgrammar 0.1.19, huggingface-hub 0.33.0, setuptools 80.9.0.
- How the lock was made: the authors built the venvs with `b3/scripts/setup_sgl.sh` (shipped): uv resolution of
  `python[srt]` with `--exclude-newer 2025-06-13` (merge date + 2 days), the fix constrained to the parent's freeze.
  Their result is `env/6994_{parent,fix}_freeze.txt`. On 2026-11-24 that freeze (minus the sglang line) was compiled
  with `uv pip compile --no-deps --generate-hashes --exclude-newer 2025-06-13 --python-version 3.10`.
- Override: **flashinfer-python 0.2.6.post1 is left out** (`env/sgl_overrides.txt`, marker "never"). It is an sdist
  that needs nvcc and the authors' pods have no CUDA toolkit. The task uses the triton attention backend and pytorch
  sampling, which do not import it. The SGLang build itself installs no dependencies (`--no-deps`); its build backend
  (setuptools) is fetched by uv under the same date cutoff.
- Checks done by `setup_lock.sh`: `uv pip freeze` equals the authors' freeze (except the local path of the sglang
  line); all installed `sglang/` files (non-`.pyc`) hash exactly as `env/6994_{parent,fix}_sgl.sha256`, the authors'
  records; `import sglang, torch` works. Files that differ between parent and fix: `env/6994_sgl_pkg_files.txt`.

## Model and LoRA adapters

- `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (the authors' snapshot), whole
  repository (3.1 GB), bf16. `model.safetensors` sha256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
  `get_model.sh` writes `refs/main` = that revision, because the engine resolves the model offline through it.
- Two random-weight PEFT LoRA adapters made locally by `b3/bench/make_lora.py <dir> 1536 8960 28 1536 256 16 <seed>`
  (r 16, alpha 32, all 7 projections, B ~ N(0, 0.002)) with the parent venv's torch, seeds 0 (`qwen15_r16`) and 1
  (`qwen15_r16b`). Their `adapter_model.safetensors` sha256 must equal the authors':
  `dd56e35acf175a153ea56a27ea7211581507371a1dbdcd6b4d4a74e656d42dfa` and
  `bfdc7cfa2f9cc67d9141c4c71da9f062ea0cedb3afcccc8908ab46b46f224b89`.

## Hardware and host software used for the card

- RTX 4090 24 GB, driver with CUDA 12.6 support (torch 2.7.1+cu126); Threadripper PRO 5975WX; one CCD (8 cores + SMT
  siblings) per GPU job via `taskset` (`b3/scripts/b3_cpus.sh`: main pod GPU0 0-7,32-39, GPU1 8-15,40-47, GPU2
  16-23,48-55, GPU3 24-31,56-63; n1 pod GPU0 0-7,32-39, GPU1 24-31,56-63). Elsewhere set `CPUS` or run unpinned.
- Venvs, model, adapters and compile caches staged on local NVMe (`$STAGE`); the persistent copies were on NFS.
- Rebuild times seen on the pods: venvs ~7 min, model and adapters < 1 min, staging ~3-4 min.
