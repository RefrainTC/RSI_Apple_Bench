# Task: make n-gram speculative decoding faster in vLLM V1

`repo/vllm` is vLLM 0.7.3.dev215 (V1 engine). `WORKLOAD.md` describes the fixed offline workload you are scored on:
Qwen2.5-1.5B with n-gram ("prompt lookup") speculative decoding, k = 5, on long prompts.

Goal: lower the workload's wall time by changing vLLM's Python code under `repo/`.

Rules
- The generated tokens must stay exactly the same, and so must the speculative proposals: for every context the
  n-gram proposer must propose the same tokens as now (the first earlier occurrence of the last n tokens, and the
  k tokens after it). The scorer checks outputs, proposals, acceptances and step counts.
- Do not change engine settings (speculation, k, n, compilation, CUDA graphs, memory), and do not edit the locked
  files (configs, argument parsing, the engine front end and core scheduler, sampling params).
- No custom GPU code (CUDA, Triton, ...). CPU JIT with numba (installed: numba 0.60.0) is allowed; numba's GPU
  targets, cfunc and ctypes are not.
- Test with `bash dev_run.sh [--baseline] [--correct] [VAR=VALUE ...]` (allowed variables: N IN OL KIND REPS).
