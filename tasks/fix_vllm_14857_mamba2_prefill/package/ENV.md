# Environment spec for #14857

Everything `b3/scripts/setup_lock.sh` and `get_model.sh` need is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/14857 ("[Model] RE: Mamba2 Prefill Performance Tweaks"), merged 2025-03-21T02:21:08Z |
| Fix (merge commit) | `296f927f2493908984707354e3cc5d7b2e41650b` (vLLM 0.8.2.dev42) |
| Parent (first parent) | `0032903a5bb7c7c655f52f4efdfcc221947e9ca8` (vLLM 0.8.2.dev41) |
| Changed by the PR | `vllm/model_executor/layers/mamba/mamba_mixer2.py` (and `_version.py` in the wheel) |

## vLLM wheels (per-commit builds from wheels.vllm.ai)

| Role | URL | sha256 |
|---|---|---|
| parent | https://wheels.vllm.ai/0032903a5bb7c7c655f52f4efdfcc221947e9ca8/vllm-0.8.2.dev41%2Bg0032903a-cp38-abi3-manylinux1_x86_64.whl | `3f21ecba40df07b29f4e8f5aa3292c0653aa3b9098c758e392f515ba9c866a50` |
| fix | https://wheels.vllm.ai/296f927f2493908984707354e3cc5d7b2e41650b/vllm-0.8.2.dev42%2Bg296f927f-cp38-abi3-manylinux1_x86_64.whl | `9a05575bb9eb32e9ab8399331aec28d9e3fdfdf5f8fd7cf85ec91a5c0c967ee7` |

The hashes are those of the files the authors downloaded on 2026-11-24 (`b3/setup/wheels/*/SOURCE.txt` on their pod).

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython, `UV_PYTHON_PREFERENCE=only-managed`).
- `env/14857_deps.lock`: 131 pinned packages with sha256 hashes, identical for parent and fix (the authors pinned the fix
  venv to the parent's freeze, so only the vLLM wheel differs). Main pins: torch 2.6.0 (cu124 wheels from PyPI),
  triton 3.2.0, transformers (see the lock), xformers, `xgrammar==0.1.17`, setuptools 77.0.3.
- How the lock was made: the authors built the venvs with `b3/scripts/setup_b3.sh` (shipped): uv resolution of the
  wheel's requirements with `--exclude-newer 2025-03-23` (merge date + 2 days), the fix constrained to the parent's
  freeze. Their result is `env/14857_{parent,fix}_freeze.txt`. On 2026-11-24 that freeze (minus the vLLM line) was
  compiled with `uv pip compile --no-deps --generate-hashes --exclude-newer 2025-03-23
  --exclude-newer-package xgrammar=2026-11-24 --python-version 3.10`, which adds the hashes of every file PyPI had for
  each pinned version by the cutoff.
- Overrides:
  - `xgrammar==0.1.17` instead of 0.1.16, which vLLM 0.8.2 pins but which was deleted from PyPI; exempt from the date
    cutoff. Without it the parent install fails (this was the acceptance test's main build problem).
  - `pyairports` is not installed; a two-line stub package is written into site-packages instead (old releases were
    deleted from PyPI; `outlines` imports it). Same stub as the authors'.
- Checks done by `setup_lock.sh`: `uv pip freeze` equals the authors' freeze (except the local path of the vLLM
  line); all installed `vllm/` files (non-`.pyc`) hash exactly as `env/14857_{parent,fix}_vllm.sha256`, the authors'
  records; `import vllm, torch` works. Files that differ between parent and fix: `env/14857_vllm_pkg_files.txt`.

## Model

- `ibm-ai-platform/Bamba-9B`, revision `50ec8766310eae056e05e42ba70df45645912e1e` (the authors' snapshot), public.
- Files: `*.safetensors *.json *.txt *.model tokenizer*` (as the authors' `dl.sh`), 19.6 GB bf16; quantised to FP8 at
  load (`quantization="fp8"` in `b3/env/14857_*.env`). Shard sha256 (= LFS blob names) are in `env/build.env`.
- `get_model.sh` writes `refs/main` = that revision, because the engine resolves the model offline through it.

## Hardware and host software used for the card

- RTX 4090 24 GB, driver with CUDA 12.x; Threadripper PRO 5975WX; one CCD (8 cores + SMT siblings) per GPU job via
  `taskset` (`b3/scripts/b3_cpus.sh`: main pod GPU0 0-7,32-39, GPU1 8-15,40-47, GPU2 16-23,48-55, GPU3 24-31,56-63;
  n1 pod GPU0 0-7,32-39, GPU1 24-31,56-63). Elsewhere set `CPUS` or run unpinned.
- Venvs, model and compile caches staged on local NVMe (`$STAGE`); the persistent copies were on NFS.
- Rebuild times seen on the pods: venvs ~8 min, model ~3 min, staging ~6-7 min.
