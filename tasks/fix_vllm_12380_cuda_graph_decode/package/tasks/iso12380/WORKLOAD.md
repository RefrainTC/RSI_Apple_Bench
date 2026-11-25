# Scoring workload (fixed)

Engine, with vLLM's defaults as `benchmarks/benchmark_latency.py` builds it at this commit (CUDA graphs ON, V0 engine):

    LLM(model="RedHatAI/Meta-Llama-3-8B-Instruct-FP8", gpu_memory_utilization=0.85, seed=0)

Requests: one batch of 8 prompts, each 32 random token ids. `SamplingParams(temperature=0, max_tokens=128,
ignore_eos=True)`: greedy, exactly 128 output tokens per request. 3 untimed warm-up `generate()` calls, then 10 timed
`generate()` calls, each on a different set of 8 prompts.

Metric: median wall time of the 10 timed `generate()` calls (s), lower is better. Hardware: one RTX 4090 (24 GB), 8 CPU
cores + their SMT siblings (one CCD).
