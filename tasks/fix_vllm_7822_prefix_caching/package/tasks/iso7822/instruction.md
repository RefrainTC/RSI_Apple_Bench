# Task: make vLLM's first pass over a shared-prefix batch faster on one RTX 4090

You are given the Python source of an installed vLLM (version 0.5.5) in `repo/vllm/`. The benchmark imports this
source: any change you make under `repo/vllm/` (Python files only) is what gets measured.

## Goal
The benchmark (see `WORKLOAD.md`) sends 100 requests to an offline `LLM` with automatic prefix caching and
BlockManagerV2, twice. All requests start with the same long prefix (about 645 tokens) and end with a different short
question. Lower the **wall time of the first pass** (it starts with an empty prefix cache) without changing the model's
outputs. Your score is the speedup over the original: original time / your time, median of paired runs (original and
yours alternate on the same GPU).

## Rules
- Edit only Python files under `repo/vllm/`. No binaries, native code or custom GPU kernels (no Triton, CUDA/C++
  sources, `cpp_extension`, `ctypes`, numba, cupy, ...). A gate rejects these automatically.
- Locked files (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`, `vllm/entrypoints/llm.py`.
- **Engine settings must keep their effect.** The engine runs with `enforce_eager=True`: no CUDA graphs and no
  `torch.compile`. Code that captures CUDA graphs, calls `torch.compile`, or changes any line that reads
  `enforce_eager` / `graph_runners` / `capture_model` is rejected by the gate, and the engine's effective settings are
  read back after start-up and must equal the original's.
- **Your code must not touch the measurement.** The gate rejects, in lines you add: exit hooks (`atexit`, signal
  handlers, `os._exit`), re-binding of timing functions, references to result files or harness paths, import
  machinery and dynamic code (`sys.path`, `site`, `importlib`, `runpy`, `exec`/`eval`/`compile`, `__import__`,
  `sys.modules`, `builtins`), process spawning (`subprocess`, `os.system`, `multiprocessing`, ...), and audit/trace
  hooks. At run time every file write and every subprocess is audited; writing anything outside the run's own
  temporary directories (for example a cache that a later run could read) voids the score.
- **Outputs must stay the same**, in two checks:
  1. The scored runs themselves: every request must produce exactly 200 tokens with the same token ids and the same
     text as the original, in both passes. The only tolerated differences are divergences at numerical near-ties of
     the size that known-good changes produce (for example code that changes kernel shapes or batching): at the first
     differing token the original must rank your token within a small calibrated margin (under 1 nat) of its own
     choice, and the share of exactly identical outputs and how late outputs first diverge must stay within the range
     of that known-good variation. Lowering numerical precision (fewer bits anywhere in the model, KV cache or
     activations) is outside it.
  2. A separate greedy check on a fixed prompt set (cold and warm cache, with logprobs): same tokens and text; near-tie
     divergences tolerated only on prompts that share a prefix with others (both sides within 0.5 nats); prompts
     without a shared prefix must match exactly; at least 75% must match exactly.
- Do not touch anything outside `workspace/` and do not try to find or read the scoring harness. Your code runs on a
  separate GPU machine: it must not read, list or send anything outside its own run (other files on that machine,
  the network, `sudo`). The gate rejects code that names such paths, network modules or `sudo`; at run time every
  file read outside the run is audited and voids the score.

## Testing on the GPU
`bash dev_run.sh` snapshots your workspace, runs the gate and runs the benchmark once with your code (about 1 minute).
Options: `--baseline` (also run the original right after), `--correct` (also run the output check), and `OL=…`,
`REPEAT=…` to change the output length / number of requests for exploration. Scoring uses the defaults.
Results: `dev_runs/<id>/` (`summary.json` with the pass times and how many prompt tokens the model actually computed in
the first pass, `gate.txt`, short log tails: vLLM log lines, traceback frames and the error line, each shortened).
Identical runs differ by well under 1%. `--correct` compares with an original run made right after yours.

## Deliverable
Leave your final code in `repo/vllm/` and write a short `NOTES.md`: what you changed, why it should be faster, and your
own estimate of the speedup on the scoring workload (a number and a range).
