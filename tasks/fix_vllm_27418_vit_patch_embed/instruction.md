# Task: Make large-image requests to a vision-language model fast

We serve Qwen2-VL-2B-Instruct from vLLM at commit `88d3141e` on one RTX 4090. Clients send one large image per
request and ask for a short answer. On the job below, each request takes longer than this GPU and these weights
need. Find what is costing the time and repair it in vLLM's Python code.

## Scoring

The job (`loads/cal.env`, run by the benchmark in `bench/`) drives vLLM's offline `LLM` with
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

You get a working directory (the *view*): the unmodified engine in a private venv, the benchmark, the load you are
scored on, the model, and `dev_bench.sh`.

```
cd <view>
cp -a venv venv_base                           # optional: an untouched copy for your own before/after timing
# edit venv/lib/python3.10/site-packages/vllm/ in place; it is your copy and nothing else shares it
venv/bin/python3 -I -m py_compile <edited files>   # catches syntax errors in seconds
./dev_bench.sh                                 # one timed run of the scored load (loads/cal.env) on venv/: ~45 s
./dev_bench.sh --venv venv_base                # the same on the untouched copy
./dev_bench.sh --load smoke                    # 1 rep, quicker
./dev_bench.sh --load cor                      # the correctness pass's prompts (outputs under runs/)
```

`dev_bench.sh` runs the scorer's job environment (clean environment, CPU pinning, offline model, same compile caches
and engine ports) and prints the scored metric (median rep, seconds), the reps and the output token counts. It
compares against nothing: time your own baseline. The GPU must be idle.

`./dev_bench.sh --load img` runs the image pass.

To submit, hand the operator your edited `venv/lib/python3.10/site-packages/vllm/` directory, or a unified diff
against `venv_base`. Only `.py` files may differ from the unmodified tree. The operator scores it with the task's
verifier: your tree is built into a fresh venv and timed against the unmodified tree in alternating runs, and the
output checks above are applied. You do not run the verifier.

## Rules

- Edit vLLM's Python code only (files under `vllm/` in your venv). No new CUDA, Triton or C++ code, and no added,
  removed or upgraded packages (torch, cuDNN and the other dependencies stay as pinned).
- Do not change `bench/`, `loads/`, `dev_bench.sh` or anything else in the view except the engine package (the scorer
  uses its own copies), and do not change engine arguments, the image processor settings or their defaults.
- Do not consult upstream history or releases newer than commit `88d3141e`.
- The GPU must be idle when you run `dev_bench.sh`.
