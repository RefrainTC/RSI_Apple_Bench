# Task: make vLLM's batch-8 generation latency lower on one RTX 4090

You are given the Python source of an installed vLLM (0.6.6.post2.dev364, V0 engine) in `repo/vllm/`. The benchmark
imports this source: any change you make under `repo/vllm/` (Python files only) is what gets measured.

## Goal
The benchmark (see `WORKLOAD.md`) runs an offline `LLM` with default settings (CUDA graphs enabled) on batches of 8
short prompts, 128 output tokens each. Lower the **median latency of the timed `generate()` calls** without changing the
model's outputs. Your score is the speedup over the original: original time / your time, median of paired runs
(original and yours alternate on the same GPU).

## Rules
- Edit only Python files under `repo/vllm/`. No binaries, native code or custom GPU kernels (no Triton, CUDA/C++
  sources, `cpp_extension`, `ctypes`, numba, cupy, ...). A gate rejects these automatically.
- Locked files (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`, `vllm/entrypoints/llm.py`.
- **Engine settings must keep their effect.** Lines that read or set `enforce_eager`, the CUDA-graph capture limits and
  sizes (`max_seq_len_to_capture`, `max_batchsize_to_capture`, `_BATCH_SIZES_TO_CAPTURE`, `cudagraph_capture_sizes`),
  `capture_model`, `torch.compile`, `num_scheduler_steps` or chunked prefill may not be added or changed. The engine's
  effective settings (including which CUDA-graph batch sizes were captured) are read back after start-up and must equal
  the original's.
- **Batch shapes must stay the same.** Every model step of every timed call is traced (how many prefill and decode tokens
  it ran); the sequence must equal the original's. Merging, splitting or skipping steps voids the score.
- **Your code must not touch the measurement.** The gate rejects, in lines you add: exit hooks, re-binding of timing
  functions, references to result files or harness paths, import machinery and dynamic code, process spawning, and
  audit/trace hooks. At run time every file write, read outside the run and subprocess is audited; the GPU's busy time
  is measured from outside the process and compared with the reported time.
- **Outputs must stay the same**: every request must produce exactly 128 tokens with the same token ids and text as the
  original. The only tolerated differences are divergences at numerical near-ties of the size that known-good changes
  produce. A separate greedy check on fixed text prompts (several batch sizes, with logprobs) must also pass.
- Do not touch anything outside `workspace/` and do not try to find or read the scoring harness, the network or `sudo`.

## Testing on the GPU
`bash dev_run.sh` snapshots your workspace, runs the gate and runs the benchmark once with your code (about 1-2 minutes).
Options: `--baseline` (also run the original), `--correct` (also run the output check), and `BS=…`, `IN=…`, `OL=…`,
`REPS=…` for exploration. Scoring uses the defaults.

## Deliverable
Leave your final code in `repo/vllm/` and write a short `NOTES.md`: what you changed, why it should be faster, and your
own estimate of the speedup on the scoring workload (a number and a range).
