# Task: Make Mamba2 prefill fast

We run the hybrid Mamba2 model Bamba-9B with vLLM 0.8.2 (V0 engine) offline on one RTX 4090, its weights quantised
to FP8 at load. The job below is dominated by prefill: many short prompts, few output tokens. Each prefill step takes
longer than this GPU and these weights need. Find what is costing the time and repair it in vLLM's Python code.

## Scoring

The job is fixed (`package/b3/env/14857_cal.env`, run by the read-only benchmark `package/b3/bench/offline_b3.py`):
`LLM(model="ibm-ai-platform/Bamba-9B", quantization="fp8", max_num_seqs=64, max_model_len=2048,
gpu_memory_utilization=0.85, seed=0)`, engine defaults otherwise (CUDA graphs for decode). Requests: 512 prompts of 64
random token ids, greedy, exactly 8 output tokens (`ignore_eos`); up to 32 sequences per prefill step. One untimed
8-prompt warm-up, then 5 timed `generate()` calls, each on a new seeded prompt set (the same seeds in both arms).
**Metric: the median wall time of the 5 calls.**

Your score is a ratio, so no absolute figure is quoted. The job runs your tree and the unmodified tree on the same GPU
in alternating order (ABBA), after 2 untimed warm-up runs, takes original time / your time for each of 3 pairs, and
scores the median. 1.0x scores nothing.

Outputs must not change, and here they can be held to exact equality, because both the original and the
maintainer's repair are bit-deterministic on this job:
- a correctness pass of 24 fixed text prompts (3 calls of batch 16, 1 and 7, 48 greedy tokens) must give the same
  token ids as the original;
- every timed rep must give the same output token counts and the same sha256 over all output token ids as the paired
  original run.

This task has no automatic gate: the analysis script prints the speedup and both checks, and the reader applies them.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it changed
or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks fixed.

## Measuring

Build and stage the package as `package/RUN.md` steps 1-4 describe (run setup with `TZ=UTC`), and run its calibration
once (step 5, the `cal` job), which also measures the maintainer's repair. The harness has no candidate argument: a
scoring job (`score`) always scores the venv staged as `$STAGE/b3/venvs/14857_fix` against
`$STAGE/b3/venvs/14857_parent`. To score your own tree, put it in that slot, then restore the repair. The commands
below follow from the shipped scripts; the authors have only scored the maintainer's repair this way, not another
tree.

```
cd package && . ./env.sh                                    # ISO_ROOT, STAGE, CPUS, PORT
V=$STAGE/b3/venvs
mv $V/14857_fix $V/14857_fix.repair                         # keep the maintainer's repair aside
cp -al $V/14857_parent $V/14857_fix                         # hard-linked copy of the unmodified tree
# edit files under $V/14857_fix/lib/python3.10/site-packages/vllm/ (see the warning below)
rm -rf $STAGE/b3/cache/s14857/fix                           # compile caches made by the repair
bash b3/scripts/b3_caldrv.sh 0 14857 score:mine1_g0         # GPU 0: 2 warm-ups, 1 correctness pass, 3 ABBA pairs
python3 -I b3/scripts/cal_analyze.py b3/cal/s14857 b3/cal/s14857/cal_g0/cor_1_parent mine1_g0
rm -rf $V/14857_fix && mv $V/14857_fix.repair $V/14857_fix  # put the repair back
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name), and delete the module's stale
`__pycache__/*.pyc`. Use a new label for every job: a job skips runs whose results already exist under its label.

One scoring job takes about 9.5 minutes of GPU time; each timed run is ~48-50 s, of which ~27-29 s is timed. Peak
memory is already 22.1 GB of 24 GB (vLLM loads bf16 weights before quantising them), and a run that crosses 97 % of
the card is stopped. `python3 -m compileall` on your edited files catches syntax errors in seconds.

## Rules

- Edit vLLM's Python code only (files under `vllm/` in your venv). No new CUDA, Triton or C++ code.
- Do not change the benchmark, the env files, the harness, engine settings or their defaults.
- Do not consult upstream history or releases newer than commit `0032903a`.
- The GPU must be idle; every run refuses to start, or aborts, if another process is on it.
