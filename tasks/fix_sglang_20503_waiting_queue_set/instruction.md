# Task: Make a large offline batch fast

We serve Qwen2.5-0.5B-Instruct from SGLang at commit `20a23e31` on one RTX 4090 and run large offline batch jobs:
tens of thousands of short requests submitted at once. On the job below, the batch takes longer than this GPU and
these weights need. Find what is costing the time and repair it in SGLang's Python code.

## Scoring

The job (`package/loads/cal.env`, run by the benchmark in `package/bench/`) drives the offline `sglang.Engine` with
triton attention, pytorch sampling, `max_running_requests` 256, decode CUDA graphs up to batch 256, radix cache off and
`mem_fraction_static` 0.80. One rep is **one `generate()` call with 16,384 requests of 32 random token ids, each
generating 16 greedy tokens** (`ignore_eos`). After an untimed 8-request warm-up, 3 reps run on fresh seeded prompts
(the same seeds in both arms), and the run's metric is the **median rep wall time**.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing; this load is noisy, so a job within 3 % of 1.0x is
reported as `no_gain`, and one job can read a few percent off. Score important candidates more than once.

Outputs must not change at all. A run scores zero if any of these fails:
- every request produces the same number of tokens as in the paired parent run;
- the live `ServerArgs` (ports aside) equal the parent's, so engine arguments and their defaults cannot change;
- every timed rep's digest (output ids and text of all 16,384 requests) equals the paired parent's;
- a correctness pass of 24 fixed text prompts (one per call, 64 greedy tokens with top-5 logprobs) is identical to the
  parent's pass in the same job.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

The package is built and staged as `package/RUN.md` steps 1-5 describe (about 13-20 minutes from an empty
directory). The harness scores a **venv**, named `20503_<name>` under `$STAGE/venvs/`, against the parent venv
`20503_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/20503_mine
cp -al $STAGE/venvs/20503_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/sglang/ (see the warning below)
$C/bin/python3 -m compileall -q <the files you edited>     # catches syntax errors in seconds
touch $STAGE/.staged/venv_20503_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 2 warm-ups, 2 correctness passes, 3 ABBA pairs
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name), and delete the module's stale
`__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the shipped variants.

Use a new label for every scoring job: runs that already finished under a label are skipped, not repeated. One scoring
job takes about 10 minutes; each run is ~64-69 s, of which ~31 s is engine start-up (including CUDA-graph capture up to
batch 256) and ~30 s the three timed reps. The summary is printed at the end and written to
`results/<label>/verdict.json`: the pairs, the speedup, each check, and the verdict (`fail:<checks>`, `no_gain`,
`below_ref`, `ties_ref`, `above_ref`).

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages.
- Do not change the benchmark, the loads, the harness or `harness/gate.json`, and do not change engine settings
  (including the scheduling policy and the number of running requests) or their defaults.
- The package was written for human readers and names the maintainer's repair in several places. Do not open
  `package/RUN.md` beyond steps 1-5, `package/ENV.md`, `package/PROVENANCE.md`, `package/env/`, `package/variants/`,
  the comments in `package/harness/`, the fix venv `$STAGE/venvs/20503_fix`, or this task's `README.md` and
  `solution/`.
- Do not consult upstream history or releases newer than commit `20a23e31`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.
