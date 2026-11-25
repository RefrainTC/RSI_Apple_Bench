# Task: Recover decode speed when clients ask for logprobs

We serve Qwen2.5-1.5B-Instruct from SGLang at commit `e2b8463c` on one RTX 4090. Our clients ask for the
log-probability of every generated token and the top-5 alternatives at every step. On the job below this engine
generates tokens more slowly than this GPU and these weights deliver. Find what is costing the time and repair it in
SGLang's Python code.

## Scoring

The job (`loads/cal.env`, run by the read-only benchmark in `bench/`) is one offline
`sglang.Engine.generate()` call of 8 requests, each 128 random prompt token ids -> 512 greedy tokens (`ignore_eos`),
every request with `return_logprob=True, top_logprobs_num=5`. The overlap scheduler and decode CUDA graphs are on
(the defaults). **Scored metric: the median wall time of 5 such calls**, each with new seeded prompts (the same seeds
in both arms), after one untimed warm-up call.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing, and a gain inside the parent's own run-to-run
noise (0.7 %) is `no_gain`.

Outputs must not change. A run scores zero if any of these fails:
- **exact**: in every timed rep, the returned token ids, text, sampled-token logprobs and top-5 logprobs (rounded to
  1e-3) equal the paired parent run's;
- every request produces the same number of tokens;
- the live engine settings (`ServerArgs`, ports aside) equal the parent's, so engine arguments and their defaults
  cannot change;
- a correctness pass of 24 fixed word prompts, sent as token ids in calls of 16, 1 and 7 requests, 64 greedy tokens
  with top-5 logprobs and logprobs for 4 fixed token ids: ids identical to the parent's, every returned logprob within
  1e-3 nats, and a top-k id may differ only where its value ties the k-th value within 1e-3.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

You get a working directory (the *view*): the unmodified engine in a private venv, the benchmark, the load you are
scored on, the model, and `dev_bench.sh`.

```
cd <view>
cp -a venv venv_base                           # optional: an untouched copy for your own before/after timing
# edit venv/lib/python3.10/site-packages/sglang/ in place; it is your copy and nothing else shares it
venv/bin/python3 -I -m py_compile <edited files>   # catches syntax errors in seconds
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~60 s
./dev_bench.sh --venv venv_base                # the same on the untouched copy
./dev_bench.sh --load smoke                    # 1 rep, quicker
./dev_bench.sh --load cor                      # the correctness pass's prompts (outputs under runs/)
```

`dev_bench.sh` runs the scorer's job environment (clean environment, CPU pinning, offline model, same compile caches
and engine ports) and prints the scored metric (median rep, seconds), the reps and the output token counts. It
compares against nothing: time your own baseline. The GPU must be idle.

To submit, hand the operator your edited `venv/lib/python3.10/site-packages/sglang/` directory, or a unified diff
against `venv_base`. Only `.py` files may differ from the unmodified tree. The operator scores it with the task's
verifier: your tree is built into a fresh venv and timed against the unmodified tree in alternating runs, and the
output checks above are applied. You do not run the verifier.

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new CUDA, Triton or C++ code.
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change engine settings or their defaults.
- Do not consult upstream history or releases newer than commit `e2b8463c`.
- The GPU must be idle when you run `dev_bench.sh`.
