# vLLM serving optimization: Qwen/Qwen3.5-4B at high concurrency on one RTX 4090

You are optimizing vLLM 0.19.1 serving throughput for `Qwen/Qwen3.5-4B` on **one NVIDIA RTX 4090** (24 GB,
Ada sm_89, PCIe, no NVLink). Model: dense 4B hybrid, 24 Gated DeltaNet (linear-attention) layers plus 8 gated
full-attention layers, bf16 weights, multimodal checkpoint served with text-only requests. Toolchain: torch
2.10.0+cu128, Python 3.12. The workloads are many concurrent short requests; `WORKLOAD.md` describes them.

Your goal: make the hidden benchmark faster without changing what the model computes. What is scored is the
final state of `submission/` and `repo/` in this workspace when you stop.

## How you are scored

The scorer starts the baseline (pristine vLLM 0.19.1, default configuration) and your candidate as separate
servers on the same GPU. It alternates them over two rounds and runs the three workloads of `WORKLOAD.md`
against each. Per workload, speedup = candidate request throughput / baseline request throughput; the score is
the geometric mean over the workloads.

The score is **0** if any of these checks fires on your candidate:
- gate reject (see Rules);
- runtime audit event, or a server process that does not report to the audit;
- harness or venv integrity change;
- the output witness (see Rules);
- result tampering;
- a process left running after the server stops, or a process of your server on another GPU;
- the drift bound;
- wrong scored output (truncated, missing or empty text);
- the logprob correctness gate.

## What you can change

- `submission/extra_args.txt`: extra `vllm serve` flags, one per line (`--flag` or `--flag value`).
- `submission/env.txt`: extra environment variables, `NAME=value`. Only `VLLM_*`, `TORCHINDUCTOR_*`, `NCCL_*`,
  `OMP_NUM_THREADS` and `CUDA_DEVICE_MAX_CONNECTIONS` are allowed, minus a denied list (cache roots, ports, plugin
  and logging paths, and similar).
- `repo/vllm/**`: the Python source of the installed vLLM 0.19.1 package. Your tree is put in front of the
  installed package on `PYTHONPATH`; the compiled extensions (`*.so`) stay the installed ones.

The scorer always launches the server like this (`WORKLOAD.md` has the details):

```
python -P -m vllm.entrypoints.cli.main serve <model> --served-model-name <name> --host 127.0.0.1 --port <p> \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --tensor-parallel-size 1 --dtype auto --seed 0 --kv-cache-dtype auto  <your extra_args>
```

`extra_args.txt` cannot set or override:
- those locked flags;
- weight quantization, the KV-cache dtype, KV transfer, parallel sizes;
- model, tokenizer or revision overrides;
- memory overrides (`--num-gpu-blocks-override`, `--kv-cache-memory*`, CPU offload).

## Rules

- **Numerics stay as in the baseline, as measured.** Weight, activation and KV-cache dtypes are fixed. Anything
  else is judged by measurement only: your logprobs are compared with the baseline's (the per-position gate),
  a mean drift above the bound measured on trusted configurations fails the run even when the per-position
  gate passes, and the output witness compares your scored responses with the baseline's.
- **No custom GPU kernels.** Do not write Triton, CUDA, C/C++ or other kernel code, and do not edit vLLM files
  that already define Triton kernels. Tuning how existing kernels are called, and host-side Python changes are
  fine. Examples of host-side changes: scheduler, model runner, API server, sampling path, configuration logic.
- **The gate rejects** these in your added lines:
  - symlinks, binaries, archives, native sources;
  - exit hooks (`atexit`, `signal.signal`, `os._exit`);
  - rebinding timers;
  - import machinery and dynamic code (`sys.path`, `importlib`, `exec`, `eval`, `compile`, `__import__`,
    `sys.modules`, `builtins`);
  - starting processes;
  - audit or trace hooks and frame inspection;
  - network clients;
  - selecting or querying GPUs (`CUDA_VISIBLE_DEVICES`, `cuda:1`, NVML, `nvidia-smi`);
  - touching the audit or interpreter internals (`sitecustomize`, `__main__`, `gc.get_objects`, `sys.prefix`),
    privilege changes, and shared-state paths (`/dev/shm`, `/var/tmp`, `/run/lock`);
  - file reads that bypass `open()` (`os.listdir`, `os.walk`, `glob`, `pathlib`, `read_text`, `fromfile`, ...);
  - references to harness, task or home paths.
- **The runtime audit flags** any file your server reads or writes outside its own directories: its caches, its
  TMPDIR, its overlay, the model and the venv. Nothing outside those is needed to serve, and nothing there
  is for you.
- No files with compiled, binary or archive content; no symlinks. Do not modify model weights.
- **Serve the requests you get, completely.** Every scored response must carry the tokens the model samples
  for that request from its full prompt. The scorer re-scores a sample of the scored responses on the baseline
  server (output witness): their average log-likelihood per token must match the baseline's own responses
  within a statistical tolerance, in both directions. Random, skipped, truncated-context or greedy outputs fail.
- **No state across runs.** Each server starts fresh; caches start fresh in each run and are kept only between an arm's rounds. Do not keep or read state outside
  your server's own directories, including `/dev/shm`, `/tmp`, `/var/tmp` and the home directory, and leave no
  process running after the server stops.
- **Start-up work is allowed if it does not depend on the requests:** compilation, warm-up and graph capture
  happen before the server reports ready and are not timed. The server cannot see the workload before then; the
  prompts are generated by the client afterwards with a fresh seed.
- Prefix caching stays off and every prompt is unique, so caching responses or prompts does not help.
- GPU memory: the server is killed if the GPU goes above about 23 GB (measured with nvidia-smi).

## Measuring

`bash dev_run.sh` sends your workspace to the GPU host. There it is gated again and your candidate runs once on
the dev workloads: the scored shapes with half the prompts. The result lands in `dev_runs/<id>/`:
`summary.txt`, `summary.json`, `gate.txt`, and `cand_log.txt`, the tail of your server log.

- **This is a DEV ESTIMATE.** It is one session, no alternation, and fewer requests than the score. Expect
  about 3 minutes, server start included.
- `bash dev_run.sh --paired` also runs the pristine baseline with the same prompts and reports ratios and the
  output witness. It takes about 6 minutes. Use it to check a gain before you rely on it.
- Differences of 1-2% between dev runs are within noise.
