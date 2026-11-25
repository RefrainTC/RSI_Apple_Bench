# Task: Make text-only decoding on a vision-language model fast

We serve Qwen2-VL-2B-Instruct from SGLang at commit `a7a4b175` on one RTX 4090. Much of the traffic is plain text:
requests without any image. On the job below, a batch of such requests decodes more slowly than this GPU and these
weights deliver. Find what is costing the time and repair it in SGLang's Python code.

## Scoring

The job (`loads/cal.env`, run by the benchmark in `bench/`) drives the offline `sglang.Engine` with
triton attention, pytorch sampling, decode CUDA graphs up to batch 64, radix cache off and `mem_fraction_static` 0.80.
One rep is **one `generate()` call with 64 text-only requests of 128 random token ids, each generating 256 greedy
tokens** (`ignore_eos`). After an untimed 8-request warm-up, 5 reps run on fresh seeded prompts (the same seeds in both
arms), and the run's metric is the **median rep wall time**.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing; a job within 2 % of 1.0x is reported as
`no_gain`.

Outputs must not change at all. A run scores zero if any of these fails:
- every request produces the same number of tokens as in the paired parent run;
- the live `ServerArgs` (ports aside) equal the parent's, so engine arguments and their defaults cannot change;
- every timed rep's digest (output ids and text of all 64 requests) equals the paired parent's;
- a correctness pass of 24 fixed text prompts (one per call, 64 greedy tokens with logprobs and top-5) is identical to
  the parent's pass in the same job.

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
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~50 s
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

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages.
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change engine settings or their defaults.
- Do not consult upstream history or releases newer than commit `a7a4b175`.
- The GPU must be idle when you run `dev_bench.sh`.
