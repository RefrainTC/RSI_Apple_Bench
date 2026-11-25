# Environment spec for SGLang #9460 (M-RoPE decode positions without per-request syncs)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/sgl-project/sglang/pull/9460, merged 2025-09-03 (SGLang v0.5.2rc1 era) |
| Fix (merge commit) | `4ed9053ecffb910c87d6c0db4ef64a7c54aa099b` |
| Parent (first parent) | `5e19b159b02ae1b519ce6c3a0d1e7ff04ab9fcd5` |
| Changed by the PR | `python/sglang/srt/model_executor/forward_batch_info.py` (decode branch of `_compute_mrope_positions`: the per-request `int(self.seq_lens[i])` reads of a GPU tensor, 2 blocking syncs per request per decode step, plus a new CPU tensor and an H2D copy, are replaced by GPU tensor arithmetic `delta + seq_lens[i] - 1`) and `python/sglang/srt/layers/rotary_embedding.py` (the now unused `MRotaryEmbedding.get_next_input_positions` removed) |

## SGLang sources

SGLang has no per-commit wheels. Each role's package is built from the commit's source tarball
(`https://codeload.github.com/sgl-project/sglang/tar.gz/<commit>`), of which only `python/` is extracted and built by the
backend its `pyproject.toml` declares (setuptools; nothing else from the tree is run). GitHub does not guarantee
byte-stable tarballs, so the check is per file: every extracted file must hash as in the authors' record
`env/<role>_src.sha256` (893 files), and the file count must match.

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 148 pinned packages with sha256 hashes, the same for parent and fix. Main pins: torch 2.8.0 (cu128
  wheels from PyPI), torchvision 0.23.0, triton 3.4.0, sgl-kernel 0.3.8, flashinfer-python 0.3.0 (installed, not used by
  this load), transformers 4.56.0, xgrammar 0.1.23, huggingface-hub 0.34.4, numpy 2.2.6, pillow 11.3.0, setuptools 80.9.0.
- How the lock was made: the authors (batch 4, `SCREEN_BATCH4.md`, candidate C47) resolved `python/[srt]` of the parent
  with uv `--exclude-newer 2025-09-05` (merge date + 2 days) from PyPI only; the fix venv was a copy of the parent venv
  with only the sglang package swapped (`--no-deps`), after checking that the two `pyproject.toml` files are identical.
  Their freezes are `env/{parent,fix}_freeze.txt` (they differ only in the sglang line, `env/freeze.diff`). On
  2026-11-24 the parent freeze minus the `sglang @ file://...` line (and nothing else: `sgl-kernel` stays) was compiled
  with `uv pip compile --no-deps --generate-hashes --exclude-newer 2025-09-05 --python-version 3.10
  --python-platform x86_64-unknown-linux-gnu`.
- Checks done by `setup/setup_lock.sh`: `uv pip freeze` equals the authors' freeze (except the local path of the sglang
  line); all installed `sglang/` files (non-`.pyc`) hash exactly as `env/{parent,fix}_sgl.sha256` (876 files);
  `import sglang, torch`. Files that differ between parent and fix: `env/files_differing.txt`.
- No CUDA toolkit is needed: this SGLang version does not JIT-compile FlashInfer modules for this load (triton attention,
  pytorch sampling), and Triton brings its own `ptxas`.

## Model

`Qwen/Qwen2-VL-2B-Instruct`, revision `1860c46f12b7449d0f5b426a7474ab715342bc65` (the snapshot batch 4 used), whole
repository (4.2 GB), bf16. LFS blobs (sha256): `model-00001-of-00002.safetensors`
`994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9`, `model-00002-of-00002.safetensors`
`92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833`. `get_model.sh` writes `refs/main` = that revision,
because the engine resolves the model offline through it. The teacher-forced check loads the same snapshot with
`transformers.Qwen2VLForConditionalGeneration` from the parent venv.

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4; the cu128 torch wheels run on it), main pod of 4090-server
  (Threadripper PRO 5975WX, 4 CCDs). One CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU0 0-7,32-39,
  GPU1 8-15,40-47, GPU2 16-23,48-55, GPU3 24-31,56-63.
- Venvs, model and compile caches staged on local NVMe (`$STAGE`); the persistent copies on NFS.

## RTX 5090 (sm_120): what a run there would need (not run)

Checked on CPU by reading the wheels' fatbin sections (2026-11-24, `c5y_work/pod/scripts/archcheck.sh`):
- torch 2.8.0+cu128 carries sm_120 (`torch._C._cuda_getArchFlags()`: sm_70 ... sm_90, sm_100, sm_120). Triton 3.4.0
  generates sm_120 code (Blackwell consumer support since Triton 3.3).
- **sgl-kernel 0.3.8 has no sm_120 code**: `common_ops.abi3.so` carries SASS for sm_80/89/90 only and `flash_ops.abi3.so`
  for sm_80/86/90, with no PTX to JIT from. SGLang imports RMSNorm, SiLU-and-mul and the rotary-embedding kernel from `sgl_kernel` on
  CUDA (`layers/layernorm.py`, `activation.py`, `rotary_embedding.py`), so on a 5090 the engine would fail with "no kernel image is available". A 5090 run needs an sgl-kernel
  built for sm_120 (a later sgl-kernel release, or a source build with `TORCH_CUDA_ARCH_LIST` including 12.0) installed
  in both venvs identically, which changes the pinned environment, so it needs its own calibration.
- FlashInfer 0.3.0 is installed but not used by this load.
- Also needed: a driver for CUDA >= 12.8 (Blackwell needs R570 or newer), its own calibration (REF, A/A and the card
  offset are hardware-specific), and the 5090 host's known driver-hang issue fixed or ruled out first (stock wheels, not
  generated kernels, but the same driver).
