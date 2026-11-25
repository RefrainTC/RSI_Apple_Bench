# Scoring workload (fixed)

Engine (vLLM V1, `VLLM_USE_V1=1`, default multiprocessing: the engine core runs in its own process):

    LLM(model="Qwen/Qwen2.5-1.5B-Instruct", speculative_model="[ngram]", num_speculative_tokens=5,
        ngram_prompt_lookup_max=4, ngram_prompt_lookup_min=3, max_model_len=4096, gpu_memory_utilization=0.80, seed=0)

Requests: 32 prompts of 2048 token ids; each prompt is its own random 40-token segment repeated to 2048 tokens.
`SamplingParams(temperature=0, max_tokens=128, ignore_eos=True)`: greedy, exactly 128 output tokens per request.

One run: an untimed warm-up rep on other prompts, then 6 reps; each rep uses a new prompt set, first prefilled with an
untimed `max_tokens=1` call (the prompts go into the prefix cache), then the timed `generate()` call with 128 tokens.

Metric: sum of the 6 timed `generate()` walls (s), lower is better. Hardware: one RTX 4090 (24 GB), 8 CPU cores +
their SMT siblings (one CCD).
