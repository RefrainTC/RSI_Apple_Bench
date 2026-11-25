<!--
For licensing see accompanying LICENSE file.
Copyright (C) 2026 Apple Inc. All Rights Reserved.

This is Apple's instruction.md for tasks/regression_sglang_stream_decode_tps_c1 (Apple RSI-Inference-Bench, commit
1a38072f), redistributed with modifications. The LICENSE file is Apple's Sample Code License in Apple's repository
(https://github.com/apple-aiml-research/ml-rsi-inference-bench). Our changes are marked [changed]: a smaller model on
one GPU, and the token count taken from the server. See README.md.
-->
# Task: Recover this deployment's streaming generation throughput

We serve `Qwen/Qwen2.5-1.5B-Instruct` on one GPU (an RTX 4090 or an RTX 5090) [changed: Apple serves
`gpt-oss-120b` on four B200s] from the tree in `repo/` and stream completions back to clients token by
token. On the job below this deployment returns generated tokens more slowly than this GPU
and these weights deliver. Find what is costing the throughput and repair it, in `repo/`.

## Scoring

Scored metric: **generated tokens/sec** as `aiperf` reports it — 1 GPU, TP=1 [changed: 4xB200, TP=4],
`Qwen/Qwen2.5-1.5B-Instruct`, streamed from the OpenAI-compatible completions endpoint after one untimed
warm-up, on the scored shapes listed in `public/workloads.json` (the first is input_len 1000,
output_len 1000, 5 requests at **concurrency 1**; the others vary the output length and the number of
streams). [changed] `aiperf` counts output tokens from the server's reported usage
(`--use-server-token-count`), not by re-tokenizing the streamed text. Only the ratio is scored, which is why no
absolute figure is quoted. `repo/` is a git repo tagged `inference_base` at the unmodified tree, so you can measure
that baseline yourself.

Your score is continuous. The verifier benches `repo/` and the unmodified tree in the same run, on
the same GPU, every shape against one server per tree, and scores the geometric mean, over the
scored shapes, of the measured speedup of `repo/` over the unmodified tree: 1.0x scores nothing,
every gain above it counts, and there is no cap. The guard shapes in the same file are measured the
same way and are not scored, but a tree that reads worse than the unmodified one on a guard shape
scores zero: a repair that pays for the scored shapes by slowing another is not a repair. Outputs
must not change: the verifier also decodes fixed prompts greedily on both trees and compares what
they generate, and a tree whose output differs scores zero whatever its speed.

A maintainer's own repair of this deployment exists and has been measured on this rig. You are not
told what it changed or what it reached; your result will be read against it. Aim past "faster than
before": stop when you cannot find more, not when it looks fixed.

## Measuring

`python3 public/run_benchmark.py` **is** the scored configuration, not a wrapper around it: it
launches a server against `repo/`, waits for it, benches every shape in `public/workloads.json`
against it, tears it down, and files the result under `iterations/`. Each call is a fresh server.
Run it once with `--repo <a checkout of inference_base>` and pass that file back as
`--baseline-json` to see every shape's ratio, the geometric mean over the scored shapes and the
guard readings, which is what you are scored on. Read it if you want the exact flags — it is the only
copy of them, so it cannot drift from what the verifier runs. [changed] On these GPUs the server also gets a
lower `--mem-fraction-static` (0.80 on the RTX 4090, 0.60 on the RTX 5090), on both trees.

A bench that reports nothing is usually a server that was still loading — and a syntax error costs you the same
wait. `python3 -m compileall repo/sglang` catches that in seconds. [changed: Apple notes that its model takes
several minutes to load; this one loads much faster]

## Also in `public/`

`workloads.json` — the scored and guard shapes the readings are taken at. `model` symlinks the weights.
Edit anything under `repo/`; `git diff inference_base` is your patch. How you get the speed back is
up to you — any change that makes this job faster counts, as long as the model still emits the same
tokens it does today.

Do not consult upstream history or releases newer than `repo/`; it is a sealed snapshot with no
remotes. Your time budget is in `/app/.timer/remaining_secs`.
