# Reference solution: vLLM #30056

- PR: https://github.com/vllm-project/vllm/pull/30056, merged 2025-12-09.
- Parent (the task's baseline): `67475a6e81abea915857f82e6f10d80b03b842c9` (vLLM 0.11.2.dev690).
- Fix (the merge commit): `c72ea1072313b30fc8b4701697520fdd0b56fc35` (vLLM 0.11.2.dev691).

## What the fix changes

One line in `vllm/v1/structured_output/__init__.py`, plus a new 5-line method in the reasoning parsers
(`vllm/reasoning/abs_reasoning_parsers.py`, `basic_parsers.py`, `deepseek_v3_reasoning_parser.py`,
`holo2_reasoning_parser.py`, `identity_reasoning_parser.py`), and the generated `vllm/_version.py`.

When a request has structured output and the engine runs a reasoning parser, the grammar may constrain only the
answer, after the model's reasoning ends (`</think>` for Qwen3). On every step, for every such request whose reasoning
has not ended yet, the engine core's `StructuredOutputManager.should_advance` asks the parser whether it has ended now.
- The parent calls `is_reasoning_end(request.all_token_ids)`: a Python loop backwards over the whole sequence, prompt
  and output, until it finds a think token. The list is a `ConstantList`, so every element access is a Python call.
- The fix calls a new `is_reasoning_end_streaming(all_ids, new_ids)`, which looks only at the tokens added in this
  step.

The answer is the same, so outputs are bit-identical. On this task's load (64 requests, sequences of 1,024-1,536 tokens,
reasoning that never ends) the parent's scan costs ~11.8 ms per step; removing it gives 1.79x per rep.

## How the package builds the fixed arm

Nothing is compiled. Both venvs come from vLLM's per-commit wheels on `wheels.vllm.ai` (URLs and sha256 in
`package/env/build.env`). The parent venv is the parent wheel plus a hashed lock of its dependencies, resolved as of
the merge date + 2 days (`package/env/deps.lock`). The fix venv is a copy of the parent's with only the vllm wheel
swapped (`--no-deps`); both wheels declare the same requirements. `setup/setup_lock.sh` checks both freezes and every
vllm file against the authors' hashes; `package/env/files_differing.txt` lists the 7 files that differ.

To score the reference solution: `bash harness/run.sh s1_g0 score` (the candidate defaults to `fix`). If it reads
`rerun:exact`, repeat it once under a new label.
