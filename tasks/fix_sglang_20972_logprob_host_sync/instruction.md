# Task: Recover decode speed when clients ask for logprobs

We serve Qwen2.5-1.5B-Instruct from SGLang at commit `e2b8463c` on one RTX 4090. Our clients ask for the
log-probability of every generated token and the top-5 alternatives at every step. On the job below this engine
generates tokens more slowly than this GPU and these weights deliver. Find what is costing the time and repair it in
SGLang's Python code.

## Scoring

The job (`package/loads/cal.env`, run by the read-only benchmark in `package/bench/`) is one offline
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

The package is built and staged as `package/RUN.md` steps 1-4 describe (about 25 minutes from an empty directory).
The harness scores a **venv**, named `20972_<name>` under `$STAGE/venvs/`, against the parent venv
`20972_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/20972_mine
cp -al $STAGE/venvs/20972_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/sglang/ (see the warning below)
touch $STAGE/.staged/venv_20972_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 2 warm-ups, 2 correctness passes, 3 pairs
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name, or `cp f f.new && rm f && mv f.new f`
before editing), and delete the module's stale `__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the
shipped variants.

One scoring job takes about 10 minutes of GPU time. Each timed run is ~60 s: ~33 s engine start-up, an 11 s warm-up
call and 5 timed calls of under 3 s each. The summary is printed at the end and written to
`results/<label>/verdict.json`: the pairs, the speedup, each check, and the verdict (`fail:<checks>`, `no_gain`,
`below_ref`, `ties_ref`, `above_ref`). `python3 -m compileall` on your edited files catches syntax errors in seconds.

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new CUDA, Triton or C++ code.
- Do not change the benchmark, the loads, the harness or `harness/gate.json`, and do not change engine settings or
  their defaults.
- Do not consult upstream history or releases newer than commit `e2b8463c`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.
