# Environment spec for #14778

Everything `b3/scripts/setup_lock.sh` and `get_model.sh` need is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/14778 ("[Model] Mamba2 Prefill Performance Tweaks"), merged 2025-03-14T20:36:18Z; reverted later and re-landed as #14857 |
| Fix (merge commit) | `fe66b34728e5d383e3d19aefc544eeee808c99fb` (vLLM 0.7.4.dev451) |
| Parent (first parent) | `270a5da495d24e947a71e2fa0c56635f4fad2dc3` (vLLM 0.7.4.dev450) |
| Changed by the PR | `vllm/model_executor/layers/mamba/mamba_mixer2.py` (the wheels also differ in `_version.py` and `_C.abi3.so`, a rebuild of unchanged sources) |

## vLLM wheels (per-commit builds from wheels.vllm.ai)

| Role | URL | sha256 |
|---|---|---|
| parent | https://wheels.vllm.ai/270a5da495d24e947a71e2fa0c56635f4fad2dc3/vllm-0.7.4.dev450%2Bg270a5da4-cp38-abi3-manylinux1_x86_64.whl | `9819f5aa286ae443c69f908a18cc65426fcf765495d9ef4692acd74f461b2569` |
| fix | https://wheels.vllm.ai/fe66b34728e5d383e3d19aefc544eeee808c99fb/vllm-0.7.4.dev451%2Bgfe66b347-cp38-abi3-manylinux1_x86_64.whl | `40e5556c4fa152d064caee7b507ef7cf25da2ce3aaffa231486e47f123f96679` |

The hashes are those of the files the authors downloaded on 2026-11-24 (`b3/setup/wheels/*/SOURCE.txt` on their pod).

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython, `UV_PYTHON_PREFERENCE=only-managed`).
- `env/14778_deps.lock`: 128 pinned packages with sha256 hashes, identical for parent and fix (the authors pinned the fix
  venv to the parent's freeze, so only the vLLM wheel differs). Main pins: torch 2.5.1 (cu124 wheels from PyPI),
  triton 3.1.0, transformers 4.49.0, xformers, `xgrammar==0.1.17`, setuptools 76.0.0.
- How the lock was made: the authors built the venvs with `b3/scripts/setup_b3.sh` (shipped): uv resolution of the
  wheel's requirements with `--exclude-newer 2025-03-16` (merge date + 2 days), the fix constrained to the parent's
  freeze. Their result is `env/14778_{parent,fix}_freeze.txt`. On 2026-11-24 that freeze (minus the vLLM line) was
  compiled with `uv pip compile --no-deps --generate-hashes --exclude-newer 2025-03-16
  --exclude-newer-package xgrammar=2026-11-24 --python-version 3.10`, which adds the hashes of every file PyPI had for
  each pinned version by the cutoff.
- Overrides:
  - `xgrammar==0.1.17` instead of the release this vLLM pins, which was deleted from PyPI; exempt from the date cutoff
    (`env/uv_overrides_14778.txt`, as the authors).
  - `pyairports` is not installed; a two-line stub package is written into site-packages instead (old releases were
    deleted from PyPI; `outlines` imports it). Same stub as the authors'.
- Checks done by `setup_lock.sh`: `uv pip freeze` equals the authors' freeze (except the local path of the vLLM
  line); all installed `vllm/` files (non-`.pyc`) hash exactly as `env/14778_{parent,fix}_vllm.sha256`, the authors'
  records; `import vllm, torch` works. Files that differ between parent and fix: `env/14778_vllm_pkg_files.txt`.

## Model

- `ibm-ai-platform/Bamba-9B`, revision `50ec8766310eae056e05e42ba70df45645912e1e` (the authors' snapshot), public.
- Files: `*.json *.txt *.model tokenizer*` only. The task loads dummy weights for the first 16 of 32 layers
  (`load_format=dummy`, `hf_overrides={"num_hidden_layers": 16}` in `b3/env/14778_*.env`), so the safetensors are
  never read. The authors staged the full 19.6 GB snapshot (shared with #14857); that makes no difference to the run.
- `get_model.sh` writes `refs/main` = that revision, because the engine resolves the model offline through it.

## Hardware and host software used for the card

- RTX 4090 24 GB, driver with CUDA 12.x; Threadripper PRO 5975WX; one CCD (8 cores + SMT siblings) per GPU job via
  `taskset` (`b3/scripts/b3_cpus.sh`: main pod GPU0 0-7,32-39, GPU1 8-15,40-47, GPU2 16-23,48-55, GPU3 24-31,56-63;
  n1 pod GPU0 0-7,32-39, GPU1 24-31,56-63). Elsewhere set `CPUS` or run unpinned.
- Venvs, model and compile caches staged on local NVMe (`$STAGE`); the persistent copies were on NFS.
- Rebuild times: venvs ~6-8 min, model < 1 min, staging ~3 min.
