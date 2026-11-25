# Task: Make the first pass over a shared-prefix batch fast

We run vLLM 0.5.5 offline on one RTX 4090 with automatic prefix caching and the v2 block manager. A batch of 100
requests shares one long prefix (about 645 tokens of a table) and ends in 100 different short questions. The first
pass over that batch, which starts with an empty prefix cache, takes longer than this GPU and these weights need.
Find what is costing the time and repair it in vLLM's Python code under `repo/vllm/`.

## Scoring

The job is fixed (`WORKLOAD.md`): `LLM(model="RedHatAI/Meta-Llama-3-8B-Instruct-FP8", enforce_eager=True,
use_v2_block_manager=True, enable_prefix_caching=True)`, 100 prompts, greedy, exactly 200 output tokens each
(`ignore_eos`). `generate()` is called twice on the same prompts. **Only the wall time of the first call is scored**;
the second is recorded and checked.

Your score is a ratio, so no absolute figure is quoted. The scorer runs your tree and the unmodified tree on the same
GPU in alternating order (ABBA), after 2 untimed warm-ups on other questions, takes original time / your time for each
of 3 pairs, and scores the median. The reward is continuous, speedup over 1.0 measured against the maintainer's
repair: a tree within noise of 1.0x gets the verdict `no_gain` but can still show a small reward.

Outputs must not change beyond numerical near-ties. A job scores zero if any check fails:
- every scored request, in both passes, has exactly 200 tokens, and the same text whenever the token ids match;
- where ids differ, the original ranks your token within a small calibrated margin (under 1 nat) of its own choice,
  and the share of exactly identical outputs and the median step of the first difference stay within the range that
  known-good changes produce. Lower numerical precision anywhere (weights, KV cache, activations) is outside it;
- a separate greedy check on fixed prompts with logprobs (cold and warm cache): prompts without a shared prefix must
  match exactly, others may differ only at near-ties (0.5 nats both ways), and at least 75 % must match exactly;
- the engine's effective settings, read back after start-up, equal the original's.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it changed
or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks fixed.

## Measuring

The candidate is a **workspace directory**. `package/RUN.md` step 5 (`bash harness/build_workspace.sh
tasks/iso7822`) builds it at `package/tasks/iso7822/workspace`: `instruction.md`, `WORKLOAD.md`, `dev_run.sh` and
`repo/vllm/`, the Python source of the installed vLLM. The benchmark imports `repo/vllm/` through an overlay on the
unmodified venv, so what you change there is what is measured.

- **Work in place, in `tasks/iso7822/workspace` itself.** Its `dev_run.sh` always measures that directory: run from a
  copy, it silently measures the untouched workspace and reports no gain even for a correct fix.
- `repo/` is its own git repository whose first commit is the pristine tree. Run `git diff` **inside `repo/`** to see
  your patch: the workspace root is a separate repository that ignores `repo/`, so `git diff` there shows nothing.
- The workspace holds an older `instruction.md` (another title, no Scoring, Measuring or Rules sections). This text
  takes precedence.
- `bash dev_run.sh` snapshots the workspace, runs the gate and runs the benchmark once with your code (about 1
  minute). `--baseline` also runs the original right after; `--correct` also runs the greedy check; `OL=...` and
  `REPEAT=...` change the output length and number of requests for exploration. Scoring always uses the defaults.
  Results land in `dev_runs/<id>/` (`summary.json` with both pass times and how many prompt tokens the model actually
  computed in the first pass, `gate.txt`, short log tails). Identical runs differ by well under 1 %.
- The official score is taken by the operator from the package root after calibration (`package/RUN.md` steps 1-6):
  `bash harness/score.sh tasks/iso7822 <label> tasks/iso7822/workspace`. It checks the scorer's integrity, snapshots and gates
  the workspace, runs the correctness check, 2 warm-ups and 3 ABBA pairs, and writes
  `tasks/iso7822/results/<label>/summary.md`. One scoring job takes about 5 minutes of GPU time.

## Rules

- Edit only Python files under `repo/vllm/`. No binaries, native code or custom GPU kernels (no Triton, CUDA/C++,
  `cpp_extension`, `ctypes`, numba, cupy). A gate rejects them.
- Locked (changes are rejected): `vllm/config.py`, `vllm/engine/arg_utils.py`, `vllm/envs.py`,
  `vllm/entrypoints/llm.py`. Lines that read or set `enforce_eager`, `graph_runners` or `capture_model`, CUDA-graph
  capture and `torch.compile` may not be added or changed.
- Your code must not touch the measurement. Added lines may not contain exit hooks, re-binding of timing functions,
  references to result files or harness paths, import machinery or dynamic code, process spawning, or audit/trace
  hooks. At run time every file write, every read outside the run and every subprocess is audited, and the GPU's busy
  time is measured from outside the process and compared with the reported time. Writing a cache that a later run
  could read voids the score.
- Do not change the harness, the package or anything outside the workspace, and do not look for the scorer's files.
- Do not consult upstream history or releases newer than the pinned tree.
