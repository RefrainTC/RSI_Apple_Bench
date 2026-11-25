# Task: Make n-gram speculative decoding faster

We run vLLM 0.7.3.dev215 (V1 engine) offline on one RTX 4090 with n-gram ("prompt lookup") speculative decoding: the
drafter proposes the k = 5 tokens that followed the first earlier occurrence of the last n tokens, and the model
verifies them. On the job below decoding takes longer than this GPU and these weights need: too much time passes
between GPU steps. Find what is costing the time and repair it in vLLM's Python code under `repo/vllm/`.

## Scoring

The job is fixed. Engine: `LLM(model="Qwen/Qwen2.5-1.5B-Instruct", speculative_model="[ngram]",
num_speculative_tokens=5, ngram_prompt_lookup_max=4, ngram_prompt_lookup_min=3, max_model_len=4096,
gpu_memory_utilization=0.80, seed=0)`, with the engine core in the same process as the driver
(`VLLM_ENABLE_V1_MULTIPROCESSING=0`, so batches are reproducible; `WORKLOAD.md` still describes the default
multiprocessing mode, and the scored runner overrides it). Requests: 32 prompts of 2048 token ids, each its own random
40-token segment repeated; greedy, exactly 128 tokens each (`ignore_eos`). One run: an untimed warm-up rep on other
prompts, then 6 reps, each on a new prompt set that is first prefilled untimed (`max_tokens=1`), then decoded with 128
tokens. **Metric: the sum of the 6 timed decode calls.**

Your score is a ratio, so no absolute figure is quoted. The scorer runs your tree and the unmodified tree on the same
GPU in alternating order (ABBA), after 2 untimed warm-up jobs, takes original time / your time for each of 3 pairs,
and scores the median. 1.0x scores nothing.

Outputs and speculation must not change. A job scores zero if any check fails:
- in every timed run, every request has exactly 128 tokens, and token ids and text hashes equal the original's;
- engine-core counters per `generate()` call equal the original's: steps, scheduled, proposed and accepted draft
  tokens, non-empty proposals, longest proposal, tokens-per-step histogram, and a hash of every proposal. For every
  context the drafter must propose the same tokens as now;
- a correctness run over several prompt groups (repeated, random, late match, short prompts over 4 ids, and a group
  whose only match lies far back in a long context) must match the original in ids, text and counters;
- a direct test of the drafter's `propose(context, n, k)` on 400 fixed contexts must give the original's results
  (when a candidate removes that API, the two checks above decide);
- the engine's effective settings (speculation, drafter, k, n, CUDA graphs, compilation level) equal the original's.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it changed
or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks fixed.

## Measuring

The candidate is a **workspace directory**: `instruction.md`, `WORKLOAD.md`, `dev_run.sh` and `repo/vllm/`, the
Python source of the installed vLLM. The benchmark imports `repo/vllm/` through an overlay on the unmodified venv, so
what you change there is what is measured. `repo/` is a git repository whose first commit is the pristine tree.

- `bash dev_run.sh [--baseline] [--correct] [VAR=VALUE ...]` snapshots the workspace, runs the gate and runs the
  benchmark once with your code; allowed variables are `N`, `IN`, `OL`, `KIND`, `REPS`. Scoring always uses the
  defaults.
- The official score is taken by the operator from the package root after calibration (`package/RUN.md` steps 1-6):
  `bash ops_iso13365/score_ws.sh <label> <workspace dir> $GPU`, which runs `harness_iso13365/score.sh` on that
  workspace: integrity check, gate, correctness run, 2 warm-ups, 3 ABBA pairs; results in
  `tasks/iso13365/results/<label>/summary.md`. One scoring job takes about 10 minutes of GPU time.

The saving, if any, is CPU time between GPU steps, so CPU contention on the machine changes the readings: compare with a
`--baseline` run from the same session.

## Rules

- Edit only Python files under `repo/vllm/`. No custom GPU code (CUDA, Triton, `cpp_extension`, ...). A CPU JIT with
  numba (numba 0.60.0 is installed in both arms) is allowed; numba's GPU targets (`numba.cuda`, `cuda.jit`,
  `target='cuda'`), `cfunc`, `intrinsic`, `numba.core` / `extending`, ctypes and cffi are not.
- Locked (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`, `vllm/entrypoints/`,
  `vllm/v1/engine/`, `vllm/v1/core/`, `vllm/v1/outputs.py`, `vllm/v1/request.py`, `vllm/sampling_params.py`,
  `vllm/utils.py`. Do not change engine settings (speculation, k, n, compilation, CUDA graphs, memory).
- Your code must not touch the measurement. Added lines may not contain exit hooks, re-binding of timing functions,
  references to result files or harness paths, import machinery or dynamic code, process spawning, or audit/trace
  hooks. At run time file writes, reads outside the run and subprocesses are audited, and the GPU's busy time is
  measured from outside the process.
- Do not change the harness, the package or anything outside the workspace, and do not look for the scorer's files.
- Do not consult upstream history or releases newer than the pinned tree.
