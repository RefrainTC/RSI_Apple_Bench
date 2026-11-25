# Workloads and measurement

## Hardware and software

| | |
|---|---|
| GPU | 1x NVIDIA RTX 4090, 24 GB GDDR6X (~1 TB/s), sm_89, PCIe 4.0, no NVLink |
| CPU | a fixed 24-core slice for the server; the benchmark client has its own 8 cores |
| Software | vLLM 0.19.1 (PyPI build), torch 2.10.0+cu128, transformers 5.9.0, Python 3.12 |
| Model | `Qwen/Qwen3.5-4B`, bf16, text-only requests (the vision tower is loaded by default) |

Model shape:
- 32 layers, hidden size 2560, MLP 9216, vocabulary 248,320 (tied embeddings);
- full attention every 4th layer: 16 query heads, 4 KV heads, head dim 256;
- the other 24 layers are Gated DeltaNet: 32 value heads and 16 key heads, head dim 128. The baseline stores
  the recurrent state in float32.

## Server

```
python -P -m vllm.entrypoints.cli.main serve <model> --served-model-name <name> --host 127.0.0.1 --port <p> \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --tensor-parallel-size 1 --dtype auto --seed 0 --kv-cache-dtype auto  <your extra_args>
```

- The environment is clean: your `env.txt` is added to it.
- Each arm (baseline, candidate) starts a scoring with fresh caches: a copy of a warm Triton/inductor cache plus an
  empty vLLM cache. The arm's round-2 server reuses what its round-1 server compiled.
  Compile time is not measured, but the server must be ready within 20 minutes.

## Client

Requests come from `vllm bench serve`, the stock vLLM 0.19.1 client, which never loads your code:

```
--backend vllm --endpoint /v1/completions --dataset-name random --random-input-len I --random-output-len O
--random-range-ratio 0 --num-prompts N --max-concurrency C --request-rate inf --ignore-eos
--temperature 1.0 --top-p 1.0 --top-k 0 --seed <fresh>
```

- **Prompts.** Random token ids, decoded to text. Every request in a scoring run is unique, and the seed
  changes every round. Both arms of a round get the same prompts.
- **Outputs.** Streaming completions with `ignore_eos`, so every request generates exactly O tokens.
- **Closed loop.** C requests are in flight at any time; a new one starts when one finishes.
- **Metric.** Request throughput: completed requests divided by the wall time of the run.

## Scored workloads

| name | input / output tokens | concurrency |
|---|---|---|
| hc_1024_64 | 1024 / 64 | 128 |
| hc_128_128 | 128 / 128 | 256 |
| hc_2048_256 | 2048 / 256 | 64 |

- **Prompt counts.** Each run uses several hundred to about a thousand requests; the exact counts are not
  published.
- **Warmup.** Each server session runs one short warmup per workload, with different prompts, before the timed
  runs. The warmup is not scored.
- **Scoring.** Two rounds, alternating baseline/candidate and candidate/baseline, with a fresh server per arm.
  - per workload: speedup = mean candidate throughput / mean baseline throughput;
  - score: the geometric mean over the three workloads;
  - the scorer may also report other request mixes, unpublished, to check that a gain is not tuned to these
    exact shapes.
- **Dev runs.** Same shapes, half the requests, one session.

## Checks (all on the candidate; any one scores 0)

| check | what it compares |
|---|---|
| logprob gate | Before measuring, fixed token sequences are teacher-forced through your server. At least 95% of positions need \|lp - lp_baseline\| <= 0.05 + 0.05 \|lp_baseline\|. |
| drift bound | The mean \|lp - lp_baseline\| over those positions must stay under a bound measured on trusted configurations. |
| output check | For every scored request: it completed without error, the reported length is exactly O, the text is not empty, and the re-tokenized length is consistent with O. |
| output witness | 32 responses per scored run (evenly spread) are teacher-forced through the baseline server with their prompts. Per workload, the mean log-probability per generated token of your responses is compared with the baseline's own responses of the same run (two-sided z-test). A sample that cannot be re-scored counts as a failure. |
| runtime audit | Every server process reports to the audit when it starts, and records file reads and writes outside its own directories, code loaded from elsewhere, started processes, and audit or trace hooks. A session with fewer than two reporting server processes fails. |
| field check | Before the final score: no process of an earlier run is left, and no new files in `/dev/shm`, `/var/tmp`, `/run/lock`, or the top level of `/tmp` and the home directory. |
| result integrity | The client reports its numbers to the scorer through a pipe. Result files that change afterwards are flagged. |
| lingering process | Any process carrying the session's tag after the server is stopped. |
| other GPU | Any process of the session holding a context on a GPU other than the assigned one. |
| integrity | The venv, model, harness, hidden files and the warm cache must be unchanged after each session. |
| GPU memory | The server is killed above about 23 GB. |
