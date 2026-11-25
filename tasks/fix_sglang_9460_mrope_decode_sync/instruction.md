# Task: Recover batched decode speed for a vision-language model

We serve Qwen2-VL-2B-Instruct from SGLang at commit `5e19b159` (v0.5.2rc1 era) on one RTX 4090. Clients send an
image and a short question and read back a long description. On the job below this engine generates tokens more
slowly than this GPU and these weights deliver. Find what is costing the time and repair it in SGLang's Python code.

## Scoring

The job (`loads/cal.env`, run by the read-only benchmark in `bench/`) is one offline
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
  (compared with the parent's pass in the same job, or with a reference pass the scorer recorded);
- teacher-forced check: the unmodified model, run in Hugging Face transformers, re-reads every output you produced
  (timed and correctness pass). Every token must be within 3.0 nats of its top choice, and in each run at most 5e-4 of
  the tokens may be more than 0.5 nats below it.

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
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~45 s
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
- Do not consult upstream history or releases newer than commit `5e19b159`.
- The GPU must be idle when you run `dev_bench.sh`.
