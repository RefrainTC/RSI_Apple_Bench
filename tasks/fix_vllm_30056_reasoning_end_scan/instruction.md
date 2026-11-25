# Task: Make structured output with a reasoning parser fast

We serve Qwen3-0.6B from vLLM at commit `67475a6e` on one RTX 4090, with the `qwen3` reasoning parser and a JSON schema
as structured output on every request. The grammar applies to the answer after the model's reasoning. On the job
below, generation is much slower than this GPU and these weights deliver. Find what is costing the time and repair it
in vLLM's Python code.

## Scoring

The job (`loads/cal.env`, run by the benchmark in `bench/`) drives vLLM's offline `LLM` with
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

You get a working directory (the *view*): the unmodified engine in a private venv, the benchmark, the load you are
scored on, the model, and `dev_bench.sh`.

```
cd <view>
cp -a venv venv_base                           # optional: an untouched copy for your own before/after timing
# edit venv/lib/python3.10/site-packages/vllm/ in place; it is your copy and nothing else shares it
venv/bin/python3 -I -m py_compile <edited files>   # catches syntax errors in seconds
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~65 s
./dev_bench.sh --venv venv_base                # the same on the untouched copy
./dev_bench.sh --load smoke                    # 1 rep, quicker
./dev_bench.sh --load cor                      # the correctness pass's prompts (outputs under runs/)
```

`dev_bench.sh` runs the scorer's job environment (clean environment, CPU pinning, offline model, same compile caches
and engine ports) and prints the scored metric (median rep, seconds), the reps and the output token counts. It
compares against nothing: time your own baseline. The GPU must be idle.

To submit, hand the operator your edited `venv/lib/python3.10/site-packages/vllm/` directory, or a unified diff
against `venv_base`. Only `.py` files may differ from the unmodified tree. The operator scores it with the task's
verifier: your tree is built into a fresh venv and timed against the unmodified tree in alternating runs, and the
output checks above are applied. You do not run the verifier.

## Rules

- Edit vLLM's Python code only (files under `vllm/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages.
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change engine arguments, the sampling parameters, the reasoning parser, the
  structured-output backend or their defaults.
- Do not consult upstream history or releases newer than commit `67475a6e`.
- The GPU must be idle when you run `dev_bench.sh`.
