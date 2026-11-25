# Task: Make batch-1 speculative decoding as fast as this GPU allows

We serve Qwen3.5-4B (a hybrid Gated-DeltaNet + attention model with one MTP layer) from vLLM at commit `4beebfd1` on
one RTX 4090, with MTP speculative decoding proposing one draft token per step. On the job below, a single request
decodes more slowly than this GPU and these weights deliver. Find what is costing the time and repair it in vLLM's
Python code.

## Scoring

The job (`package/loads/cal.env`, run by the benchmark in `package/bench/`) drives vLLM's offline `LLM` with
`speculative_config={"method": "qwen3_5_mtp", "num_speculative_tokens": 1}`, prefix caching off,
`max_model_len` 4096 and `gpu_memory_utilization` 0.80. One rep is **one request of 256 random token ids that
generates 512 greedy tokens** (`ignore_eos`). After an untimed warm-up, 3 reps run on fresh seeded prompts (the same
seeds in both arms), and the run's metric is the **median rep wall time**.

Your score is a ratio, so no absolute figure is quoted. The harness runs your tree and the unmodified tree on the
same GPU in alternating order (candidate, parent, parent, candidate, candidate, parent), takes parent time / your time
for each of the 3 pairs, and scores the median. 1.0x scores nothing. The effect sizes here are small, so a job within
1 % of 1.0x is reported as `no_gain`.

Outputs must not change at all. A run scores zero if any of these fails:
- every timed rep produces the same number of tokens as the paired parent run;
- vLLM's resolved configuration (`vllm_config`, including `speculative_config`) equals the parent's, so engine
  arguments and their defaults cannot change;
- every timed rep's output ids hash to the same digest as the paired parent's;
- a correctness pass of 24 fixed natural-text prompts (one per call, 64 greedy tokens, with the chosen token's
  logprob) gives the same ids as the parent's pass in the same job, and every logprob within 1e-3 nats.

The timed prompts are random ids, and on them the model's output is very repetitive. The correctness pass is natural
text, so it exercises speculative decoding the way real prompts do. Both must hold.

A maintainer's own repair of this problem exists and has been measured on this rig. You are not told what it
changed or what it reached; your result will be read against it. Stop when you cannot find more, not when it looks
fixed.

## Measuring

The package is built and staged as `package/RUN.md` steps 1-4 describe (about 12-20 minutes from an empty
directory). The harness scores a **venv**, named `35442_<name>` under `$STAGE/venvs/`, against the parent venv
`35442_parent`. To make and score your own:

```
cd package && . ./env.sh                                   # sets ISO_ROOT, STAGE, GPU, CPUS, PORT from site.env
C=$STAGE/venvs/35442_mine
cp -al $STAGE/venvs/35442_parent $C                        # hard-linked copy of the unmodified tree
# edit files under $C/lib/python3.10/site-packages/vllm/ (see the warning below)
$C/bin/python3 -m compileall -q <the files you edited>     # catches syntax errors in seconds
touch $STAGE/.staged/venv_35442_mine                       # marks it as staged
bash harness/run.sh mine1_g0 score mine                    # 2 warm-ups, 2 correctness passes, 3 ABBA pairs
```

The copy shares every file with the parent venv. **Never write into a file in place**: that changes the baseline too.
Replace the file instead (write a new file and `mv` it over the old name), and delete the module's stale
`__pycache__/*.pyc`. This is what `harness/mkvariant.sh` does for the shipped variants.

Use a new label for every scoring job: runs that already finished under a label are skipped, not repeated. Every
candidate run starts from a fresh copy of the parent's compile cache on that GPU, so both arms run the same tuned
kernels. One scoring job takes about 13-16 minutes; each run is ~75 s, of which ~57 s is engine start-up
(torch.compile, CUDA-graph capture) and ~11 s the three timed reps. The summary is printed at the end and written to
`results/<label>/verdict.json`: the pairs, the speedup, each check, and the verdict (`fail:<checks>`, `no_gain`,
`below_ref`, `ties_ref`, `above_ref`).

## Rules

- Edit vLLM's Python code only (files under `vllm/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages.
- Do not change the benchmark, the loads, the harness or `harness/gate.json`, and do not change engine arguments,
  the speculative configuration or their defaults.
- The package was written for human readers and names the maintainer's repair in several places. Do not open
  `package/RUN.md` beyond steps 1-4, `package/ENV.md`, `package/PROVENANCE.md`, `package/env/`, `package/variants/`,
  the comments in `package/harness/`, the fix venv `$STAGE/venvs/35442_fix`, or this task's `README.md` and
  `solution/`.
- Do not consult upstream history or releases newer than commit `4beebfd1`.
- The GPU must be idle; the harness refuses to start, or aborts, if another process is on it.
