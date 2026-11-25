# Task: Serve mixed bursts of requests faster with vLLM, without changing the outputs

This is a summary of the instruction an agent sees. The agent's own copy is Phi-Bench's `instruction.md` for
`vllm-scheduler-mixed-batch-serving` (Apache-2.0), unchanged except that it names the GPU it runs on; that file
still says "H20" in our runs.

You have a vLLM 0.10.1.1 OpenAI-compatible server for a fixed instruction-tuned model (bfloat16, at `/app/model`) on
one GPU: an RTX 5090 (32 GB) or an RTX 4090 (24 GB). Under concurrent bursts of dissimilar requests (short
interactive requests, long prefills and long decode streams arriving together) it finishes the burst later than this
GPU allows. Make it serve faster without changing what it generates.

## Scoring

```
speedup = median over the held-out burst workloads of ( baseline_time / your_time )
reward  = min(1, ln(speedup / REF) / ln(REF))  if speedup > REF, else 0
```

- The baseline is a tuned vLLM launch (CUDA graphs, chunked prefill, prefix caching, a tuned token budget and
  admission width). It is measured before and after your server, and each pair uses the mean of the two.
- `REF` is a constant recalibrated for the GPU you run on. Tying it scores 0; `REF^1.5` scores 0.5; `REF^2` scores 1.
- **Correctness first.** Your server and the baseline answer the same hidden prompts greedily (temperature 0), and the
  outputs are compared token by token, including degenerate and adversarial prompts. Below the required match rate
  the score is 0, whatever the speed.
- **Latency gate.** Sequential single-request latency is measured too and must not become materially worse than the
  baseline's; failing it scores 0.
- On these GPUs the baseline and every candidate run with `--gpu-memory-utilization 0.85` (the original 0.88 does
  not leave enough headroom on these cards).

## Measuring

The verifier starts your server with `/app/submission/launch_server.sh`, which receives `PORT` and `MODEL_PATH` and
must bring up an OpenAI-compatible server on `PORT`. Anything needed at verification time must live under
`/app/submission/` and be applied by `launch_server.sh` at start-up. `/app/run_dev_bench.py` is a public dev
benchmark with different workloads and fewer iterations; a good dev number does not guarantee the score. One full
verifier run takes about 21-25 minutes.

## Rules

- Edit only `/app/submission/` (it may patch vLLM at launch from files kept there). Do not touch the tests, the
  reference outputs or the reward manifest; the verifier scans the submission for references to them.
- Outputs must stay token-identical under greedy decoding.
- Do not consult upstream history or releases newer than the installed vLLM.
