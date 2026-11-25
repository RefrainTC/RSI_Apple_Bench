# Task: Recover small-batch decode speed for Qwen3.5

We serve Qwen3.5-4B from SGLang at commit `09e9c4fd` on one RTX 4090, mostly to a handful of concurrent streams. On
the job below this engine generates tokens a little more slowly than this GPU and these weights deliver. The gap is a
few percent, and the scorer is quiet enough to see it. Find what is costing the time and repair it in SGLang's Python
code.

## Scoring

The job (`package/loads/cal.env`, run by the read-only benchmark in `package/bench/`) is one offline
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
  `harness/ref_correct.json`).

Any change of numerics fails this gate, even a harmless one.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

The package is built and staged as `package/RUN.md` steps 1-5 describe (about 20 minutes from an empty directory).
The harness scores a **venv**, named `27868_<name>` under `$STAGE/venvs/`, against the parent venv `27868_parent`.
To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/27868_mine
cp -al $STAGE/venvs/27868_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/sglang/ (see the warning below)
touch $STAGE/.staged/venv_27868_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 2 warm-ups, 2 correctness passes, 3 pairs
bash harness/run.sh try1_g0 quick mine                     # cheaper check (~5 min): 1 pair; not a score
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name, or `cp f f.new && rm f && mv f.new f`
before editing), and delete the module's stale `__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the
shipped variants.

One scoring job takes about 9-11 minutes of GPU time. Each run is ~56 s: ~27 s engine start-up including CUDA-graph
capture, the warm-up, and 3 timed calls of ~6 s. The summary is printed at the end and written to
`results/<label>/verdict.json`: the pairs, the speedup, each check, and the verdict (`fail:<checks>`, `no_gain`,
`below_ref`, `ties_ref`, `above_ref`). `python3 -m compileall` on your edited files catches syntax errors in seconds.

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new GPU kernels.
- Do not change the benchmark, the loads, the harness, `harness/gate.json` or `harness/ref_correct.json`, and do not
  change engine settings or their defaults.
- Do not consult upstream history or releases newer than commit `09e9c4fd`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.
