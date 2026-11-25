# Task: Make LoRA decoding fast

We serve Qwen2.5-1.5B-Instruct with a LoRA adapter from SGLang 0.4.7 (offline `sglang.Engine`) on one RTX 4090.
Decode runs at batch 64 inside CUDA graphs. On the job below every decode step takes several milliseconds longer
than this GPU, these weights and this adapter need. Find what is costing the time and repair it in SGLang's Python
code.

## Scoring

The job is fixed (`package/b3/env/6994_cal.env`, run by the read-only benchmark `package/b3/bench/sgl_b3.py`):
`sglang.Engine` with the triton attention and LoRA backends, pytorch sampling, radix cache off, CUDA graphs up to
batch 64, `max_loras_per_batch` 4, `mem_fraction_static` 0.80, seed 0; adapters `lora0` and `lora1` (rank 16, all 7
projections). Requests: 64 prompts of 256 random token ids, every request on `lora0`, greedy, exactly 256 output
tokens (`ignore_eos`). One untimed 8-request warm-up, then 5 timed `generate()` calls, each on a new seeded prompt set
(the same seeds in both arms). **Metric: the median wall time of the 5 calls.**

Your score is a ratio, so no absolute figure is quoted. The job runs your tree and the unmodified tree on the same GPU
in alternating order (ABBA), after 2 untimed warm-up runs, takes original time / your time for each of 3 pairs, and
scores the median. 1.0x scores nothing.

Outputs must not change:
- a correctness pass of 24 fixed text prompts (3 calls of batch 16, 1 and 7, 48 greedy tokens, top-5 logprobs) must
  give the same token ids as the original (the maintainer's repair reproduces them exactly);
- every timed rep must produce the same number of output tokens per request as the paired original run.

The output text of timed reps is not bit-reproducible even between two runs of the original (SGLang groups the
arriving requests by timing), so it is reported, not required. This task has no automatic gate: the analysis script
prints the speedup and both checks, and the reader applies them.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it changed
or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks fixed.

## Measuring

Build and stage the package as `package/RUN.md` steps 1-4 describe, and run its calibration once (step 5, the `cal`
job), which also measures the maintainer's repair. The harness has no candidate argument: a scoring job (`score`)
always scores the venv staged as `$STAGE/b3/venvs/6994_fix` against `$STAGE/b3/venvs/6994_parent`. To score your own
tree, put it in that slot, then restore the repair. The commands below follow from the shipped scripts; the authors
have only scored the maintainer's repair this way, not another tree.

```
cd package && . ./env.sh                                    # ISO_ROOT, STAGE, CPUS, PORT
V=$STAGE/b3/venvs
mv $V/6994_fix $V/6994_fix.repair                           # keep the maintainer's repair aside
cp -al $V/6994_parent $V/6994_fix                           # hard-linked copy of the unmodified tree
# edit files under $V/6994_fix/lib/python3.10/site-packages/sglang/ (see the warning below)
rm -rf $STAGE/b3/cache/s6994/fix                            # compile caches made by the repair
bash b3/scripts/b3_caldrv.sh 0 6994 score:mine1_g0          # GPU 0: 2 warm-ups, 1 correctness pass, 3 ABBA pairs
python3 -I b3/scripts/cal_analyze.py b3/cal/s6994 b3/cal/s6994/cal_g0/cor_1_parent mine1_g0
rm -rf $V/6994_fix && mv $V/6994_fix.repair $V/6994_fix     # put the repair back
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name), and delete the module's stale
`__pycache__/*.pyc`. Use a new label for every job: a job skips runs whose results already exist under its label.

One scoring job takes about 4.5 minutes of GPU time; each timed run is ~29-35 s, of which ~12-18 s is timed. Progress
is in `b3/cal/s6994/<label>/cal.log`. `python3 -m compileall` on your edited files catches syntax errors in seconds.

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new CUDA, Triton or C++ code.
- Do not change the benchmark, the env files, the adapters, the harness or engine settings and their defaults.
- Do not consult upstream history or releases newer than commit `777688b8`.
- The GPU must be idle; every run refuses to start, or aborts, if another process is on it.
