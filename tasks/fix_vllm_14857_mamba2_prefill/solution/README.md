# Reference solution: vLLM #14857 (and the duplicate #14778)

- Upstream PR: https://github.com/vllm-project/vllm/pull/14857 ("[Model] RE: Mamba2 Prefill Performance Tweaks: Fixing
  Flurry of Unnecessary Memory Copies"), merged 2025-03-21.
- Fix (merge commit): `296f927f2493908984707354e3cc5d7b2e41650b` (vLLM 0.8.2.dev42).
- Parent (first parent, the task's baseline): `0032903a5bb7c7c655f52f4efdfcc221947e9ca8` (0.8.2.dev41).

## What it changes

One file, 6 lines: `vllm/model_executor/layers/mamba/mamba_mixer2.py`. In `MambaMixer2.forward_cuda`'s prefill branch:
- the zero-initialisation of fresh SSM states, a loop of `ssm_state[idx].zero_()` over a CUDA index tensor behind a
  Python `any()` over a CUDA tensor, becomes `torch.any` plus one indexed assignment;
- the write-back of final states, a loop of `ssm_state[idx].copy_(varlen_state[i])`, becomes one indexed assignment.

Each loop iteration indexed with a 0-d CUDA tensor, a GPU->CPU sync per sequence per Mamba layer per prefill step.
Outputs are bit-identical.

## The duplicate: #14778

https://github.com/vllm-project/vllm/pull/14778, fix `fe66b34728e5d383e3d19aefc544eeee808c99fb`, parent
`270a5da495d24e947a71e2fa0c56635f4fad2dc3` (vLLM 0.7.4.dev, merged 2025-03-14). Same parent code; the fix keeps
`torch.any` but replaces both loops with `torch.vmap` over a lambda that indexes `ssm_state[idx]` and writes in place.
It was reverted upstream and re-landed as #14857. Its wheels also differ in `_C.abi3.so` (a rebuild; no C++/CUDA change
in the PR). Packaged as `iso14778` on a dummy-weight, 16-layer bf16 load: REF 1.0365.

## How the package builds the fixed arm

- Both arms are vLLM's per-commit wheels from wheels.vllm.ai (URLs and sha256 in `package/ENV.md`), installed from one
  hashed lock (`package/env/14857_deps.lock`, 131 packages, identical for both arms) with the `xgrammar==0.1.17`
  override. `package/b3/scripts/setup_lock.sh` checks the freeze and all 1012 installed `vllm/` files against the
  authors' build; only `mamba_mixer2.py` and `_version.py` differ between the arms. Run it with `TZ=UTC`.
- The fix arm is that whole venv (no overlay): `b3/scripts/b3_cal.sh` runs role `fix` from `$STAGE/b3/venvs/14857_fix`.

## Result

REF **1.0648** on RTX 4090 (7 jobs on 4 cards and two servers, 1.0614-1.0656, band ±0.5 %); outputs identical to the
parent's in every rep. See `../README.md` and `../results/summary.json`.
