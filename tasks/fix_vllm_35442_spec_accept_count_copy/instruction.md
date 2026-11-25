# Task: Make batch-1 speculative decoding as fast as this GPU allows

We serve Qwen3.5-4B (a hybrid Gated-DeltaNet + attention model with one MTP layer) from vLLM at commit `4beebfd1` on
one RTX 4090, with MTP speculative decoding proposing one draft token per step. On the job below, a single request
decodes more slowly than this GPU and these weights deliver. Find what is costing the time and repair it in vLLM's
Python code.

## Scoring

The job (`loads/cal.env`, run by the benchmark in `bench/`) drives vLLM's offline `LLM` with
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

You get a working directory (the *view*): the unmodified engine in a private venv, the benchmark, the load you are
scored on, the model, and `dev_bench.sh`.

```
cd <view>
cp -a venv venv_base                           # optional: an untouched copy for your own before/after timing
# edit venv/lib/python3.10/site-packages/vllm/ in place; it is your copy and nothing else shares it
venv/bin/python3 -I -m py_compile <edited files>   # catches syntax errors in seconds
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~75 s
./dev_bench.sh --venv venv_base                # the same on the untouched copy
./dev_bench.sh --load smoke                    # 1 rep, quicker
./dev_bench.sh --load cor                      # the correctness pass's prompts (outputs under runs/)
```

`dev_bench.sh` runs the scorer's job environment (clean environment, CPU pinning, offline model, same compile caches
and engine ports) and prints the scored metric (median rep, seconds), the reps and the output token counts. It
compares against nothing: time your own baseline. The GPU must be idle.

To submit, hand the operator your edited `venv/lib/python3.10/site-packages/vllm/` directory, or a unified diff
against `venv_base`. Only `.py` files may differ from the unmodified tree. The operator scores it with the task's
verifier: your tree is built into a fresh venv and timed against the unmodified tree in alternating runs, and the
output checks above are applied. You do not run the verifier.

## Rules

- Edit vLLM's Python code only (files under `vllm/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages.
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change engine arguments, the speculative configuration or their defaults.
- Do not consult upstream history or releases newer than commit `4beebfd1`.
- The GPU must be idle when you run `dev_bench.sh`.
