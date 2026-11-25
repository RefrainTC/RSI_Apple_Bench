# Task: Bring this deployment's batch-8 generation latency back down

We run vLLM 0.6.6.post2.dev364 (V0 engine) offline on one RTX 4090, with the engine's defaults, CUDA graphs included.
On the job below each `generate()` call takes much longer than this GPU and these weights deliver: the decode steps
cost more than they should. Find what is costing the time and repair it in vLLM's Python code under `repo/vllm/`.

## Scoring

The job is fixed (`WORKLOAD.md`): `LLM(model="RedHatAI/Meta-Llama-3-8B-Instruct-FP8", gpu_memory_utilization=0.85,
seed=0)`, one batch of 8 prompts of 32 random token ids, greedy, exactly 128 output tokens each (`ignore_eos`). 3
untimed warm-up calls, then 10 timed `generate()` calls, each on a different prompt set. **Metric: the median wall
time of the 10 timed calls.**

Your score is a ratio, so no absolute figure is quoted. The scorer runs your tree and the unmodified tree on the same
GPU in alternating order (ABBA), after 2 untimed warm-up jobs, takes original time / your time for each of 3 pairs,
and scores the median. 1.0x scores nothing.

Outputs must not change. A job scores zero if any check fails:
- every scored request produces exactly 128 tokens with the same token ids and text as the original. The tolerance
  is calibrated on known-good code, which reproduces the original bit for bit, so in effect the match must be exact;
- a separate greedy check on 25 fixed text prompts (several batch sizes, with logprobs) passes;
- every model step of every timed call is traced (how many prefill and decode tokens it ran), and the sequence must
  equal the original's. Merging, splitting or skipping steps voids the score;
- the engine's effective settings, including which CUDA-graph batch sizes were captured, are read back after start-up
  and must equal the original's.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it changed
or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks fixed.

## Measuring

The candidate is a **workspace directory**: `instruction.md`, `WORKLOAD.md`, `dev_run.sh` and `repo/vllm/`, the
Python source of the installed vLLM. The benchmark imports `repo/vllm/` through an overlay on the unmodified venv, so
what you change there is what is measured. `repo/` is a git repository whose first commit is the pristine tree, so
`git diff` is your patch.

- `bash dev_run.sh` snapshots the workspace, runs the gate and runs the benchmark once with your code (about 1-2
  minutes). `--baseline` also runs the original; `--correct` also runs the greedy check; `BS=...`, `IN=...`, `OL=...`,
  `REPS=...` change the load for exploration. Scoring always uses the defaults.
- The official score is taken by the operator from the package root after calibration (`package/RUN.md` steps 1-6):
  `VERDICT_BAND=0.011 bash harness_iso12380/score.sh tasks/iso12380 <label> <workspace dir>`. It checks the scorer's
  integrity, snapshots and gates the workspace, runs the correctness check, 2 warm-ups and 3 ABBA pairs, and writes
  `tasks/iso12380/results/<label>/summary.md`. One scoring job takes about 9 minutes of GPU time.

The unmodified side is CPU-bound and sensitive to other work on the machine: compare against a `--baseline` run from
the same session.

## Rules

- Edit only Python files under `repo/vllm/`. No binaries, native code or custom GPU kernels (no Triton, CUDA/C++,
  `cpp_extension`, `ctypes`, numba, cupy). A gate rejects them.
- Locked (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`,
  `vllm/entrypoints/llm.py`.
- Engine settings must keep their effect. Lines that read or set `enforce_eager`, the CUDA-graph capture limits and
  sizes (`max_seq_len_to_capture`, `max_batchsize_to_capture`, `_BATCH_SIZES_TO_CAPTURE`, `cudagraph_capture_sizes`),
  `capture_model`, `torch.compile`, `num_scheduler_steps` or chunked prefill may not be added or changed.
- Your code must not touch the measurement. Added lines may not contain exit hooks, re-binding of timing functions,
  references to result files or harness paths, import machinery or dynamic code, process spawning, or audit/trace
  hooks. At run time every file write, every read outside the run and every subprocess is audited, and the GPU's busy
  time is measured from outside the process and compared with the reported time.
- Do not change the harness, the package or anything outside the workspace, and do not look for the scorer's files.
- Do not consult upstream history or releases newer than the pinned tree.
