# Task: Recover batched decode speed for a vision-language model

We serve Qwen2-VL-2B-Instruct from SGLang at commit `5e19b159` (v0.5.2rc1 era) on one RTX 4090. Clients send an
image and a short question and read back a long description. On the job below this engine generates tokens more
slowly than this GPU and these weights deliver. Find what is costing the time and repair it in SGLang's Python code.

## Scoring

The job (`package/loads/cal.env`, run by the read-only benchmark in `package/bench/`) is one offline
`Engine.generate()` call of 64 requests. Each request is one synthetic 224 x 224 PNG plus the fixed prompt "Describe
this image in detail.", 256 greedy output tokens (`ignore_eos`). Triton attention, pytorch sampling, radix cache off,
CUDA graphs up to batch 64. **Scored metric: the median wall time of 5 such calls**, each on a new seeded image set
(the same seeds in both arms), after one untimed 8-request warm-up.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing, and a gain inside the parent's own run-to-run
noise (3 %) is `no_gain`. Single pairs on this load vary by several percent; only the job's median counts.

Outputs must stay correct. The outputs of the 64-request calls legitimately differ from run to run, also for the
unmodified engine (requests are batched in arrival order), so they are not compared bit for bit. A run scores zero
if any of these fails:
- every request produces the same number of tokens as in the paired parent run;
- the live engine settings (`ServerArgs`, ports aside) equal the parent's, so engine arguments and their defaults
  cannot change;
- a correctness pass of 16 fixed prompts (12 with an image of 224, 336 or 448 px, 4 text-only), sent one at a time,
  64 greedy tokens: the token ids must equal the unmodified engine's exactly and every logprob must be within 1e-3
  (compared with the parent's pass in the same job, or with the recorded reference pass `harness/ref_correct.json`);
- teacher-forced check: the unmodified model, run in Hugging Face transformers, re-reads every output you produced
  (timed and correctness pass). Every token must be within 3.0 nats of its top choice, and in each run at most 5e-4 of
  the tokens may be more than 0.5 nats below it.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

The package is built and staged as `package/RUN.md` steps 1-4 describe (about 15 minutes from an empty directory;
longer on slow disks). The harness scores a **venv**, named `9460_<name>` under `$STAGE/venvs/`, against the parent
venv `9460_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/9460_mine
cp -al $STAGE/venvs/9460_parent $C                         # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/sglang/ (see the warning below)
touch $STAGE/.staged/venv_9460_mine                        # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 2 warm-ups, 2 correctness passes, 3 pairs, teacher-forced check
bash harness/run.sh try1_g0 quick mine                     # cheaper check (~4 min): 1 pair; not a score
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name, or `cp f f.new && rm f && mv f.new f`
before editing), and delete the module's stale `__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the
shipped variants.

One scoring job takes 8-9 minutes of GPU time. Each timed run is ~42-46 s, of which ~20 s is engine start-up. The
summary is printed at the end and written to `results/<label>/verdict.json`: the pairs, the speedup, each check, and
the verdict (`fail:<checks>`, `no_gain`, `below_ref`, `ties_ref`, `above_ref`). Keep the host quiet while a job runs:
the image front end runs on the CPU, and host load moves both arms. `python3 -m compileall` on your edited files
catches syntax errors in seconds.

## Rules

- Edit SGLang's Python code only (files under `sglang/` in your venv). No new GPU kernels.
- Do not change the benchmark, the loads, the harness, `harness/gate.json` or `harness/ref_correct.json`, and do not
  change engine settings or their defaults.
- Do not consult upstream history or releases newer than commit `5e19b159`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.
