# Task: Make this server's decode faster per token

We serve Qwen2.5-1.5B-Instruct with `vllm serve` (vLLM 0.8.5.dev117, V1 engine) on one RTX 4090, with `-O3`
(torch.compile and CUDA graphs) and the model's default generation config, which applies a repetition penalty of 1.1.
On the load below each generated token takes longer than this GPU and these weights need. Find what is costing the
time and repair it in vLLM's Python code under `repo/vllm/`.

## Scoring

The job is fixed (`WORKLOAD.md`). Server, running your code: `vllm serve Qwen/Qwen2.5-1.5B-Instruct
--enable-prefix-caching --dtype float16 --disable-log-requests -O3 --gpu-memory-utilization 0.85`. Client, the
original vLLM code: `vllm bench serve --request-rate 1 --num-prompts 100 --random-input-len 1000 --random-output-len
100 --ignore-eos`. **Metric: client-side median time per output token (TPOT).**

Your score is a ratio, so no absolute figure is quoted. The scorer runs your tree and the unmodified tree on the same
GPU in alternating order (ABBA), after 2 untimed warm-ups, takes original TPOT / your TPOT for each of 2 pairs, and
scores the median. The reward is continuous, speedup over 1.0 measured against the maintainer's repair: a tree within
noise of 1.0x gets the verdict `no_gain` but can still show a small reward (the untouched tree once read 1.0022x,
reward 0.024).

Outputs must not change beyond numerical near-ties. A job scores zero if any check fails:
- a correctness client sends a fixed prompt set greedily with logprobs, one by one and all at once: at least 90 % of
  outputs must match the original exactly, and the rest may differ only at near-ties (each side ranks the other's
  token within 0.5 nats of its own choice);
- every timed repeat: the same number of completions, the same output lengths, no errors, and at least 90 % of the
  texts identical to the paired original run;
- the median time to first token (TTFT) may not exceed 1.05x the paired original's. TPOT bought with TTFT does not
  count;
- the engine's effective configuration, read from the server's start-up log, equals the original's.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it changed
or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks fixed.

## Measuring

The candidate is a **workspace directory**. `package/RUN.md` step 5 (`bash harness/build_workspace.sh
tasks/iso16436`) builds it at `package/tasks/iso16436/workspace`: `instruction.md`, `WORKLOAD.md`, `dev_run.sh` and
`repo/vllm/`, the Python source of the installed vLLM. The server imports `repo/vllm/` through an overlay on the
unmodified venv, so what you change there is what is measured.

- You may work in `tasks/iso16436/workspace` or in a copy of it: its `dev_run.sh` measures the workspace it sits in.
- `repo/` is its own git repository whose first commit is the pristine tree. Run `git diff` **inside `repo/`** to see
  your patch: the workspace root is a separate repository that ignores `repo/`, so `git diff` there shows nothing.
- The workspace's `instruction.md` is this text when the package is built where it ships, as `package/` next to this
  file. If the package was copied elsewhere first, the workspace gets the package's own shorter task text instead
  (another title, no Scoring or Measuring sections); this text takes precedence.
- `bash dev_run.sh` snapshots the workspace, runs the gate, starts the server with your code and runs the load once
  (about 3-4 minutes). `--baseline` also runs the original right after (compare against it: a single run's TPOT varies
  by about ±1 %); `--correct` also runs the output check; `RATE=...`, `NP=...`, `IN=...`, `OL=...` change the load for
  exploration. Scoring always uses the defaults. Results land in `dev_runs/<id>/`.
- The official score is taken by the operator from the package root after calibration (`package/RUN.md` steps 1-6):
  `bash harness/score.sh tasks/iso16436 <label> tasks/iso16436/workspace`. It checks the scorer's integrity, snapshots and gates
  the workspace, runs the correctness client, 2 warm-ups and 2 ABBA pairs, and writes
  `tasks/iso16436/results/<label>/summary.md`. One scoring job takes about 19 minutes of GPU time.

## Rules

- Edit only Python files under `repo/vllm/`. No binaries, native code or custom GPU kernels (no Triton, CUDA/C++,
  `torch.utils.cpp_extension`, `ctypes`, numba, cupy). A gate rejects them.
- Locked (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`,
  `vllm/entrypoints/openai/cli_args.py`, `vllm/entrypoints/cli/`, `vllm/benchmarks/`,
  `vllm/transformers_utils/config.py`. Server flags and the load generator are fixed.
- Engine settings must keep their effect: lines that read or set the compilation config, CUDA-graph capture or
  `enforce_eager` may not be added or removed.
- Your code must not touch the measurement. Added lines may not contain exit hooks, re-binding of timing functions,
  references to result files or harness paths, import machinery or dynamic code, process spawning, or audit/trace
  hooks. At run time every file write, every read outside the run and every subprocess in the server is audited.
- Do not change the harness, the package or anything outside the workspace, and do not look for the scorer's files.
- Do not consult upstream history or releases newer than the pinned tree.
