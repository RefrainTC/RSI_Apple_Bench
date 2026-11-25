# Environment spec for vLLM #30056 (structured output with a reasoning parser: end-of-reasoning check on the new tokens only)

Everything `setup/*.sh` needs is in `env/`; this page says what it is and where it came from.

## Commits

| | |
|---|---|
| PR | https://github.com/vllm-project/vllm/pull/30056, merged 2025-12-09 |
| Fix (merge commit) | `c72ea1072313b30fc8b4701697520fdd0b56fc35` (vLLM 0.11.2.dev691) |
| Parent (first parent) | `67475a6e81abea915857f82e6f10d80b03b842c9` (vLLM 0.11.2.dev690) |
| Changed by the PR | `vllm/v1/structured_output/__init__.py` (1 line) and the reasoning parsers under `vllm/reasoning/` (a new method). When a request has structured output and the engine runs a reasoning parser, the grammar may only constrain the answer after the model's reasoning ends (`</think>` for Qwen3). On every step, for every such request whose reasoning has not ended yet, the engine core's `StructuredOutputManager.should_advance` asks the parser whether it has ended now. The parent calls `is_reasoning_end(request.all_token_ids)`, which walks the whole token list backwards in Python. The fix calls a new `is_reasoning_end_streaming(all_ids, new_ids)`, which looks only at the tokens added in this step. The other changed files are the same 5-line method in other parsers. |

## vLLM wheels

Per-commit wheels from `wheels.vllm.ai` (URL and sha256 in `env/build.env`; the authors' records are
`env/<role>_SOURCE.txt`). Both link CUDA 12 (`libcudart.so.12`). Nothing is compiled.

## Python and dependencies

- uv 0.11.14; interpreter `uv python install 3.10.20` (uv-managed CPython).
- `env/deps.lock`: 155 pinned packages with sha256 hashes, the same for parent and fix. Main pins: torch 2.9.0 (PyPI;
  CUDA 12.8 build), torchvision 0.24.0, torchaudio 2.9.0, triton 3.5.0, flashinfer-python 0.5.3, transformers 4.57.3,
  xgrammar 0.1.27, numpy 2.2.6 (all versions: `env/parent_freeze.txt`).
- How the lock was made:
  - The authors (batch-6 screen on main, `c5y/b6m`) installed the parent wheel with uv `--exclude-newer 2025-12-11`
    (merge date + 2 days).
  - The fix venv is a copy of the parent's with only the vllm wheel swapped (`--no-deps`). Both wheels declare the same
    requirements.
  - The freezes are `env/{parent,fix}_freeze.txt`; they differ only in the vllm line (`env/freeze.diff`).
  - On 2026-11-25 the parent freeze minus the vllm line was compiled with `uv pip compile --no-deps --generate-hashes
    --exclude-newer 2025-12-11`.
- Checks done by `setup/setup_lock.sh`:
  - `uv pip freeze` equals the authors' freeze (apart from the vllm line's path);
  - all installed `vllm/` files (non-`.pyc`, 1,673) hash exactly as `env/{parent,fix}_vllm.sha256`;
  - `import vllm, torch` works.

  Files that differ between parent and fix: `env/files_differing.txt`.

## Model

`Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` (the snapshot the batch-6 screen downloaded on
2026-11-25), whole repository (~1.5 GB), bf16. Its tokenizer has `<think>` = 151667 and `</think>` = 151668. LFS blobs
(sha256) are listed in `env/build.env` (`MODEL_LFS`).

## Engine settings of the load

`LLM(model, gpu_memory_utilization=0.80, seed=0, max_model_len=2048, max_num_seqs=64, enable_prefix_caching=False,
reasoning_parser="qwen3")`. The structured-output backend is vLLM's default for this version (xgrammar for a JSON
schema). Async scheduling is off by default at this commit, so the check runs in the engine core's serial step.
Compile caches (Triton, Inductor, vLLM's torch.compile cache) live under `$STAGE/cache/<pr>/<role>/<host>-g<gpu>`.
Every candidate run starts from a fresh copy of the parent's cache on that GPU (as in `packages/vllm35442`).

## Hardware and host software used for the card

- RTX 4090 24 GB (sm_89), driver 550.127.08 (CUDA 12.4), main pod of 4090-server (Threadripper PRO 5975WX, 4 CCDs). One
  CCD (8 cores + SMT siblings) per GPU job via `taskset`: GPU2 16-23,48-55, GPU3 24-31,56-63 (GPU0 0-7,32-39, GPU1
  8-15,40-47).
- Venvs and model staged on local NVMe (`$STAGE`); the persistent copies on NFS.

## RTX 5090 (sm_120): not run

torch 2.9.0's CUDA 12.8 build carries sm_120. The gain is engine-core Python per step, so it depends on the CPU more than
on the GPU, and would need its own calibration after the 5090 host's driver-hang issue is ruled out.
