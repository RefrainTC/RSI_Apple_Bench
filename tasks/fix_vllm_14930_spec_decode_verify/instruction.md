# Task: Make speculative decoding steps cheaper

We run vLLM 0.8.0rc3.dev21 (V1 engine) offline on one RTX 4090 with n-gram speculative decoding: each step the
drafter proposes 3 tokens from earlier text and the model verifies them, accepting some and rejecting the rest. On the
job below each engine step costs more than this GPU and these weights need. Find what is costing the time and repair
it in vLLM's Python code under `repo/vllm/`.

## Scoring

The job is fixed. Engine: `LLM(model="Qwen/Qwen2.5-1.5B-Instruct", speculative_model="[ngram]",
num_speculative_tokens=3, ngram_prompt_lookup_max=4, ngram_prompt_lookup_min=2, max_model_len=4096,
gpu_memory_utilization=0.80, seed=0)`, V1, default compilation, CUDA graphs and prefix caching. Requests: 64 prompts
of 1024 tokens per rep, each a random 40-token segment repeated, whose first copy carries 4 substituted tokens, so
every request sees rejected drafts; the prompts come from a fixed, pre-screened pool. Greedy, exactly 256 tokens each
(`ignore_eos`). One run: an untimed warm-up, then 7 reps, each on its own prompts, first prefilled by an untimed
1-token pass, then decoded. **Metric: the median of the 7 timed rep walls.** (The package's `WORKLOAD.md` describes
an earlier prompt mix; the scored runner uses the one above.)

Your score is a ratio, so no absolute figure is quoted. The scorer runs your tree and the unmodified tree on the same
GPU in alternating order (ABBA), after 2 untimed warm-up jobs, takes original time / your time for each of 3 pairs,
and scores the median. 1.0x scores nothing.

Outputs and acceptance must not change. A job scores zero if any check fails:
- in every timed rep, every request has exactly 256 tokens; token ids equal the original's for at least 95.3 % of the
  requests; the per-request acceptance sequence (tokens accepted per engine step) is identical for at least 95.3 %,
  and mean tokens per step is within ±6.1 % of the original's;
- a correctness run on 40 fixed prompts: greedy token ids equal the original's for at least 85 %, and with seeded
  sampling (temperature 0.8) the mean tokens per engine step stays within ±15 % of the original's (±24 % on the lure
  group). Verification of sampled requests must stay a correct rejection sampler;
- the engine's effective settings (speculative config and k, eager mode, prefix caching, compile level, CUDA graphs,
  attention backend) equal the original's.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it changed
or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks fixed.

## Measuring

The candidate is a **workspace directory**: `instruction.md`, `WORKLOAD.md`, `dev_run.sh` and `repo/vllm/`, the
Python source of the installed vLLM. The benchmark imports `repo/vllm/` through an overlay on the unmodified venv, so
what you change there is what is measured. `repo/` is a git repository whose first commit is the pristine tree.

- `bash dev_run.sh [--baseline] [--correct] [VAR=VALUE ...]` snapshots the workspace, runs the gate and runs the
  benchmark once with your code; allowed variables are `N` and `OL`. Scoring always uses the defaults.
- The official score is taken by the operator from the package root after calibration (`package/RUN.md` steps 1-6):
  `bash ops_iso14930/score_job.sh $GPU <label> <workspace dir>`, which runs `harness_iso14930/score.sh` on that
  workspace through the normal gate: integrity check, correctness run, 2 warm-ups, 3 ABBA pairs; results in
  `tasks/iso14930/results/<label>/summary.md`. One scoring job takes about 13 minutes.

## Rules

- Edit only Python files under `repo/vllm/`. No new GPU kernels: no Triton, CUDA, inline C++, `cpp_extension`,
  ctypes, numba, cupy. A gate rejects them, and rejects changes to files that already define Triton kernels.
- Locked (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`,
  `vllm/entrypoints/llm.py`, `vllm/v1/engine/`, `vllm/v1/spec_decode/ngram_proposer.py`. Do not change the engine
  config, the drafter, the engine core loop or the offline entry point.
- Your code must not touch the measurement. Added lines may not contain exit hooks, re-binding of timing functions,
  references to result files or harness paths, import machinery or dynamic code, process spawning, or audit/trace
  hooks. At run time file writes, reads outside the run and subprocesses are audited in every process, and the GPU's
  busy time is measured from outside.
- Do not change the harness, the package or anything outside the workspace, and do not look for the scorer's files.
- Do not consult upstream history or releases newer than the pinned tree.
