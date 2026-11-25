# Scoring workload (fixed)

Engine, exactly as vLLM's `benchmarks/benchmark_prefix_caching.py` builds it at this commit:

    LLM(model="RedHatAI/Meta-Llama-3-8B-Instruct-FP8", tokenizer_mode="auto", trust_remote_code=True,
        enforce_eager=True, use_v2_block_manager=True, enable_prefix_caching=True)   # gpu_memory_utilization capped at 0.85

Requests: 100 prompts = the benchmark's fixed ~645-token table prompt as a shared prefix + a different short question
per request ("Question i: Look at ... of the table above and write a detailed, numbered explanation ...").
`SamplingParams(temperature=0, max_tokens=200, ignore_eos=True)`: greedy, exactly 200 output tokens per request.
`llm.generate()` is called twice on the same prompts; the first call starts with an empty prefix cache.

Metric: wall time of the first `generate()` call (s), lower is better. The second call is recorded and checked, not
scored. Hardware: one RTX 4090 (24 GB), 8 CPU cores + their SMT siblings (one CCD).
