# Task: Recover small-batch decode speed for Qwen3.5

We serve Qwen3.5-4B from SGLang at commit `09e9c4fd` on one RTX 4090, mostly to a handful of concurrent streams. On
the job below this engine generates tokens a little more slowly than this GPU and these weights deliver. The gap is a
few percent, and the scorer is quiet enough to see it. Find what is costing the time and repair it in SGLang's Python
code.

## Scoring

The job (`loads/cal.env`, run by the read-only benchmark in `bench/`) is one offline
`Engine.generate()` call of 4 requests, each 256 random prompt token ids -> 512 greedy tokens (`ignore_eos`). Triton
attention, pytorch sampling, decode CUDA graphs up to batch 8. Nearly all of the time is decode at batch 4 inside CUDA
graphs. **Scored metric: the median wall time of 3 such calls**, each with new seeded prompts (the same seeds in both
arms), after one untimed 8-request warm-up.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing, and a gain of 1 % or less is `no_gain`.

Outputs must not change at all. The unmodified engine is deterministic run to run on this job, so the gate is exact.
A run scores zero if any of these fails:
- **exact**: in every timed rep, the output token ids and text equal the paired parent run's;
- every request produces the same number of tokens;
- the live engine settings (`ServerArgs`, ports aside) equal the parent's, so engine arguments and their defaults
  cannot change;
- a correctness pass of 24 fixed text prompts, one per `generate()` call, 64 greedy tokens with logprobs: ids
  identical and every logprob within 1e-3 of the parent's pass in the same job (or of the recorded reference pass
  a reference pass the scorer recorded).

Any change of numerics fails this gate, even a harmless one.

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
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~56 s
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

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new GPU kernels.
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change engine settings or their defaults.
- Do not consult upstream history or releases newer than commit `09e9c4fd`.
- The GPU must be idle when you run `dev_bench.sh`.
