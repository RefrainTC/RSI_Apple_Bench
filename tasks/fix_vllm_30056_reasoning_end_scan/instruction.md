# Task: Make structured output with a reasoning parser fast

We serve Qwen3-0.6B from vLLM at commit `67475a6e` on one RTX 4090, with the `qwen3` reasoning parser and a JSON schema
as structured output on every request. The grammar applies to the answer after the model's reasoning. On the job
below, generation is much slower than this GPU and these weights deliver. Find what is costing the time and repair it
in vLLM's Python code.

## Scoring

The job (`package/loads/cal.env`, run by the benchmark in `package/bench/`) drives vLLM's offline `LLM` with
`reasoning_parser="qwen3"`, `max_model_len` 2048, `max_num_seqs` 64, prefix caching off and
`gpu_memory_utilization` 0.80. One rep is **one `generate()` call with 64 requests of 1,024 random token ids, each
generating 512 greedy tokens** (`ignore_eos`). Every request carries the same small JSON schema (`{"answer": string}`)
as structured output, and a `logit_bias` of -100 on `<think>` and `</think>`, so no request ever finishes its reasoning
during the timed reps. After an untimed warm-up, 3 reps run on fresh seeded prompts (the same seeds in both arms), and
the run's metric is the **median rep wall time**.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing; a job within 3 % of 1.0x is reported as
`no_gain`.

Outputs must not change at all. A run scores zero if any of these fails:
- every timed rep produces the same number of tokens as the paired parent run;
- vLLM's resolved configuration (`vllm_config`, including `structured_outputs_config`) equals the parent's, so engine
  arguments, the reasoning parser and their defaults cannot change;
- every timed rep's output ids hash to the same digest as the paired parent's;
- a correctness pass of 24 short questions in Qwen3's chat format with thinking on, each with the same JSON schema and
  no `logit_bias` (one per call, 384 greedy tokens with logprobs), gives the same ids and logprobs (within 1e-3 nats) as
  the parent's pass in the same job. In this pass the model thinks, writes `</think>`, and then answers under the
  grammar, which the timed load never reaches. The summary prints how many of the 24 prompts ended their reasoning.

The timed load is not perfectly repeatable: about one run in 50, in either arm, one rep's digest differs. So if counts,
settings and the correctness pass all pass and exactly one timed rep of one run differs, the verdict is `rerun:exact`.
Run the same job once more under a new label; that rerun's verdict is final, a second `fail:exact` included.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

The package is built and staged as `package/RUN.md` steps 1-4 describe (about 12-18 minutes from an empty
directory). The harness scores a **venv**, named `30056_<name>` under `$STAGE/venvs/`, against the parent venv
`30056_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/30056_mine
cp -al $STAGE/venvs/30056_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/vllm/ (see the warning below)
$C/bin/python3 -m compileall -q <the files you edited>     # catches syntax errors in seconds
touch $STAGE/.staged/venv_30056_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 2 warm-ups, 2 correctness passes, 3 ABBA pairs
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name), and delete the module's stale
`__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the shipped variants.

Use a new label for every scoring job: runs that already finished under a label are skipped, not repeated. Every
candidate run starts from a fresh copy of the parent's compile cache on that GPU, so both arms run the same tuned
kernels. One scoring job takes about 9 minutes; a parent run is ~64 s, of which ~17 s is engine start-up and ~41 s the
three timed reps. The summary is printed at the end and written to `results/<label>/verdict.json`: the pairs, the
speedup, each check, and the verdict (`fail:<checks>`, `rerun:exact`, `no_gain`, `below_ref`, `ties_ref`,
`above_ref`).

## Rules

- Edit vLLM's Python code only (files under `vllm/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages.
- Do not change the benchmark, the loads, the harness or `harness/gate.json`, and do not change engine arguments, the
  sampling parameters, the reasoning parser, the structured-output backend or their defaults.
- The package was written for human readers and names the maintainer's repair in several places. Do not open
  `package/RUN.md` beyond steps 1-4, `package/ENV.md`, `package/PROVENANCE.md`, `package/env/`, `package/variants/`,
  the comments in `package/harness/`, the fix venv `$STAGE/venvs/30056_fix`, or this task's `README.md` and
  `solution/`.
- Do not consult upstream history or releases newer than commit `67475a6e`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.
