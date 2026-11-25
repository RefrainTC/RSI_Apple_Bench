# Task
Make vLLM V1 speculative decoding faster on the workload in WORKLOAD.md by changing engine Python code in `repo/`.
The rejection sampler (`vllm/v1/sample/rejection_sampler.py`) and the model runner's spec-decode path are the places to
look. Rules: no new GPU kernels (no Triton, CUDA, inline C++); do not change the engine config, the drafter, the engine
core loop or the offline entry point (locked); outputs must stay identical to the parent's greedy outputs, and the
number of draft tokens accepted per step must not change. Test with `bash dev_run.sh [--baseline] [--correct]`.
