# Scoring workload (fixed)

Server (your code via `repo/vllm/`):
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-prefix-caching --dtype float16 --disable-log-requests -O3 \
        --gpu-memory-utilization 0.85
(the model's default generation config is used: repetition_penalty 1.1, temperature/top-p/top-k from the model.)

Load generator (original vLLM, not your code):
    vllm bench serve --model Qwen/Qwen2.5-1.5B-Instruct --request-rate 1 --num-prompts 100 \
        --random-input-len 1000 --random-output-len 100 --ignore-eos
Metric: median TPOT (ms), lower is better. Hardware: one RTX 4090 (24 GB), 8 CPU cores + their SMT siblings (one CCD).
