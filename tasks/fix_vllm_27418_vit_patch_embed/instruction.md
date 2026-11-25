# Task: Make large-image requests to a vision-language model fast

We serve Qwen2-VL-2B-Instruct from vLLM at commit `88d3141e` on one RTX 4090. Clients send one large image per
request and ask for a short answer. On the job below, each request takes longer than this GPU and these weights
need. Find what is costing the time and repair it in vLLM's Python code.

## Scoring

The job (`package/loads/cal.env`, run by the benchmark in `package/bench/`) drives vLLM's offline `LLM` with
`max_model_len` 4096, `max_num_seqs` 32, one image per prompt, vision attention `TORCH_SDPA` and
`gpu_memory_utilization` 0.80. Every image is a synthetic 1344 x 1344 picture, seeded per rep and request (so nothing
is cached between requests), processed at exactly that size (2,304 image tokens). One rep is **4 requests, each in its
own `generate()` call, one after the other**, each "Describe this image in detail." with 8 greedy tokens. After one
untimed warm-up call, 5 reps run, and the run's metric is the **median rep wall time**.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing; this load is noisy, so a job within 5 % of 1.0x is
reported as `no_gain`.

Image outputs may differ slightly in floating point, but they must stay what the model would write. A run scores zero
if any of these fails:
- every request produces the same number of output tokens from the same prompt length (the 2,304 image tokens
  included) as the paired parent run, so the image resolution cannot change;
- vLLM's resolved configuration (`vllm_config`) equals the parent's, so engine arguments and their defaults cannot
  change;
- **teacher-forced re-read:** after the job, the unmodified engine rebuilds each image from its seed and re-reads
  everything you generated in one prefill. Every token must be within 1.0 nat of the unmodified engine's top choice,
  and at most 2 % of your tokens may be more than 0.5 nats below it. This covers the timed reps and an image pass of 8
  simple pictures with a question about their content (48 tokens each);
- a pass of 24 text-only prompts (one per call, 64 tokens with logprobs) is bit-identical to the parent's pass in the
  same job.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

The package is built and staged as `package/RUN.md` steps 1-4 describe (about 15-20 minutes from an empty
directory). The harness scores a **venv**, named `27418_<name>` under `$STAGE/venvs/`, against the parent venv
`27418_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/27418_mine
cp -al $STAGE/venvs/27418_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/vllm/ (see the warning below)
$C/bin/python3 -m compileall -q <the files you edited>     # catches syntax errors in seconds
touch $STAGE/.staged/venv_27418_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # warm-ups, text and image passes, 3 ABBA pairs, re-read
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name), and delete the module's stale
`__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the shipped variants.

Use a new label for every scoring job: runs that already finished under a label are skipped, not repeated. Every
candidate run starts from a fresh copy of the parent's compile cache on that GPU; any recompilation is part of the
untimed start-up. One scoring job takes about 11 minutes: a parent run is
~41 s, of which ~16 s is engine start-up and ~12 s the five timed reps, and the re-read adds ~2 minutes. The summary is
printed at the end and written to `results/<label>/verdict.json`: the pairs, the speedup, each check (with the
re-read's worst gap), and the verdict (`fail:<checks>`, `no_gain`, `below_ref`, `ties_ref`, `above_ref`).

## Rules

- Edit vLLM's Python code only (files under `vllm/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages (torch, cuDNN and the other dependencies stay as pinned).
- Do not change the benchmark, the loads, the harness or `harness/gate.json`, and do not change engine arguments, the
  image processor settings or their defaults.
- The package was written for human readers and names the maintainer's repair in several places. Do not open
  `package/RUN.md` beyond steps 1-4, `package/ENV.md`, `package/PROVENANCE.md`, `package/env/`, `package/variants/`,
  the comments in `package/harness/`, the fix venv `$STAGE/venvs/27418_fix`, or this task's `README.md` and
  `solution/`.
- Do not consult upstream history or releases newer than commit `88d3141e`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.
