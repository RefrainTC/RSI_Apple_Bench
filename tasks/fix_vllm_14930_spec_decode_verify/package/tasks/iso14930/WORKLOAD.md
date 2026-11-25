# Workload (#14930)
Offline `LLM.generate` with Qwen2.5-1.5B-Instruct on vLLM V1, n-gram speculative decoding (3 draft tokens per step,
n-gram 2-4). 64 prompts of 1024 tokens built from repeated 40-token segments (half of them with random substitutions in
every copy), 256 output tokens each, greedy, ignore_eos. The prefix cache is filled by an untimed 1-token pass, so the
timed part is the decode phase. Metric: median wall time of 7 timed reps (each with its own prompts). Lower is better.
Outputs must equal the parent's greedy outputs, and the number of tokens accepted per step must match the parent's.
