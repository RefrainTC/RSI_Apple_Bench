# Task: make vLLM decode faster (median TPOT) for a small chat model on one RTX 4090

You are given the Python source of an installed vLLM (version 0.8.5.dev117, V1 engine) in `repo/vllm/`. The server in the
scoring workload imports this source: any change you make under `repo/vllm/` (Python files only) is what gets measured.

## Goal
Lower the **median time per output token (TPOT)** of the scoring workload (see `WORKLOAD.md`) without changing the
model's outputs. Your score is the speedup of your version over the original: original median TPOT / your median TPOT,
measured as the median of paired runs (original and yours alternate on the same GPU).

## Rules
- Edit only Python files under `repo/vllm/`. Do not add binaries, native code or custom GPU kernels: no Triton,
  CUDA/C++ sources, `torch.utils.cpp_extension`, `ctypes`, numba, cupy and similar. A gate rejects these automatically.
- These files are locked (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`,
  `vllm/entrypoints/openai/cli_args.py`, `vllm/entrypoints/cli/`, `vllm/benchmarks/`, `vllm/transformers_utils/config.py`.
  The server flags and the load generator are fixed; the task is about the engine code, not configuration.
- Outputs must stay the same: greedy decoding with the model's default generation config (which includes a repetition
  penalty of 1.1) must produce the same tokens as the original on a fixed prompt set, sent one by one and all at once.
  The only tolerated differences are divergences at numerical near-ties (both sides rank the other's token within
  0.5 nats of their own choice); at least 90% of the outputs must match exactly.
- **Engine settings must keep their effect** (the server runs with `-O3`: torch.compile and CUDA graphs as configured).
  Lines that read or set the compilation config, CUDA graph capture or `enforce_eager` must not be changed (the gate
  checks added and removed lines); the engine's effective configuration is read from its start-up log and must equal
  the original's.
- **Your code must not touch the measurement.** The gate rejects, in lines you add: exit hooks (`atexit`, signal
  handlers, `os._exit`), re-binding of timing functions, references to result files or harness paths, import
  machinery and dynamic code (`sys.path`, `site`, `importlib`, `runpy`, `exec`/`eval`/`compile`, `__import__`,
  `sys.modules`, `builtins`), process spawning (`subprocess`, `os.system`, `multiprocessing`, ...), and audit/trace
  hooks. At run time every file write and every subprocess in the server is audited; writing anything outside the
  run's own temporary directories (for example a cache that a later run could read) voids the score.
- The scored requests' outputs are checked too: every request must complete with exactly the requested number of
  tokens, and at least 90% of the texts must equal the original's (the rest may differ only through numerical
  near-ties caused by different batching).
- Do not touch anything outside `workspace/` and do not try to find or read the scoring harness. Your code runs on a
  separate GPU machine: it must not read, list or send anything outside its own run (other files on that machine,
  the network, `sudo`). The gate rejects code that names such paths, network modules or `sudo`; at run time every
  file read outside the run is audited and voids the score.

## Testing on the GPU
`bash dev_run.sh` snapshots your workspace, runs the gate, starts the server with your code and runs the scoring
workload once (about 3–4 minutes). Options:
- `--baseline` also runs the original right after, so you get a same-session comparison (doubles the time).
- `--correct` also runs the output check against the original's reference outputs.
- `RATE=…`, `NP=…`, `IN=…`, `OL=…` change the load for exploration (request rate, number of prompts, input and output
  length). Scoring always uses the defaults in `WORKLOAD.md`.
Results appear in `dev_runs/<id>/` (`summary.json`, `status.txt`, `gate.txt`, short log tails: vLLM log lines,
traceback frames and the error line, each shortened). A single run's TPOT varies by
about ±1% between identical runs; compare against a `--baseline` run from the same session when it matters.

## Deliverable
Leave your final code in `repo/vllm/`. Write a short `NOTES.md` in the workspace: what you changed, why it should be
faster, and your own estimate of the speedup you expect on the scoring workload (a number and a range).
