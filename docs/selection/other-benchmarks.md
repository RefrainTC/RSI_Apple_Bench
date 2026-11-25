# Other benchmarks we screened

**Short answer.** Only two of Apple's 37 tasks run on our GPUs, so we looked for other inference-optimization tasks with a known correct answer. Two sources fit. **ISO-Bench** gave us real performance fixes from vLLM and SGLang history, and 8 of the 29 we measured became tasks. From **Φ-Bench** we ported one task, `vllm-scheduler-mixed-batch-serving`, because it is the only Φ-Bench task that tests an LLM inference server end to end. Its other 42 GPU tasks optimize single kernels or train models.

## ISO-Bench: real performance fixes from vLLM and SGLang

**What it is.** A list of 54 merged pull requests (39 vLLM, 15 SGLang). Each is a real performance fix, with the commit before and after. The human fix is a natural reference answer. ISO-Bench ships no harness that runs on our GPUs.

**What we did.**
1. **Paper check of all 54.** For each PR: is the model open and small enough, is there a prebuilt engine wheel for both commits, which CUDA build does it link, does it fit a 24 GB card, and does the PR's own benchmark command exercise the changed code.
2. **GPU screen of 29**, in three batches, on a 4090. See [iso-bench-screen.md](iso-bench-screen.md).

**Result.** 8 of 29 passed the screen. Two of them are the same change merged twice (#14778, reverted, then re-landed as #14857), so there are 7 independent tasks.

**The paper check was far too optimistic.** It rated 12 of the first 13 PRs as feasible, and 2 of those passed on the GPU. Overall it overestimated the usable fraction about fivefold. The paper check answers "can it run?"; it cannot answer "does the fix show on this GPU under some load?". That needs a measurement.

Reasons the paper check excluded PRs before any GPU run:
- no per-commit engine wheel (#3357, #4270, #4594, #5602);
- multi-GPU, too large, or another accelerator (#11870 70B, #21334 Llama-4, #13897 needs DeepSeek-V2-Lite in bf16, #13589 Neuron, #13305 ROCm);
- Hopper-only kernels (SGLang #5969, FA3);
- features that need several GPUs or nodes (SGLang two-batch overlap, prefill/decode split, expert load balancing, data-parallel attention);
- arrival-bound API-side paths (#17515, a reasoning parser at 1 request/s).

## Φ-Bench: why we ported the vLLM scheduler task

**Why this task.** Φ-Bench has 43 GPU tasks. Only `vllm-scheduler-mixed-batch-serving` tests an inference server:
- Qwen2.5-1.5B served by vLLM 0.10.1.1 on one GPU. The agent must make six bursts of mixed requests finish faster.
- It ships what a usable task needs: an authors' reference answer (the "oracle"), a negative example that must be rejected, and an identity arm (the baseline scored as a candidate, i.e. an A/A test).
- The model is small and open, and vLLM 0.10.1.1 runs on the 5090 once the image's base is CUDA 12.8.

**What the port changed.**
- The image's base: CUDA 12.8 with the cu128 build of torch 2.7.1, so it runs on sm_120.
- `gpu-memory-utilization` 0.88 → 0.85. At 0.88 the card peaked within 748 MiB of our 30 GiB limit on the 5090.
- Container memory 96 → 32 GB (the measured peak was 3.3 GiB).
- **A recalibration.** Apple-style tasks reward speed relative to a reference speedup measured on the authors' GPU (an H20). Kept as is, the authors' own reference answer would already score 0.21-0.26 on our 5090.

**Results.**

| | RTX 5090 | RTX 4090 |
|---|---|---|
| Authors' reference answer | 1.363 (5 runs, 1.355-1.374) | 1.393 |
| Identity arm (A/A) | 1.002 (0.996-1.005) | within ±0.5% |
| Best configuration-only answer from a parameter sweep | 1.609 | 1.712 |
| Time per scoring | 20-25 min | 20-25 min |

The negative example is rejected for the right reason (0 of 1166 outputs match). The authors' answer clears A/A by a wide margin.

**Caveats.**
- **The authors' bar is too easy on our GPUs.** A configuration-only answer reaches 1.61× (5090) and 1.71× (4090). In our own runs we used the strongest configuration as the bar, measured per card.
- **The scored load mostly rewards prefix caching.** It reuses the same long prompt many times. With unique prefixes, the configuration-only answer drops from 1.62× to 1.18×. A six-round agent run on this task climbed to 2.75× mostly by exploiting this (see [../findings.md](../findings.md)).
- **The bar depends on the host's load.** An independent tester rebuilt the task on a quieter 5090 and got 1.344 for the authors' answer and 1.55 for the configuration. The baseline is limited by the CPU-side scheduler, so it gains the most from a quiet host and the ratios shrink. See [../acceptance/batch1-5090.md](../acceptance/batch1-5090.md).
- **Problems in the task itself, independent of hardware:**
  - by its code, the reward at the reference speedup is 0, not the 0.5 its manifest documents;
  - the reference answer's `--api-server-count 4` has no effect in the way the task launches the server;
  - its scorer does not restore the baseline's environment before the second baseline measurement, which can zero a correct answer (see [../findings.md](../findings.md)).
- Source: the Φ-Bench repository, https://github.com/one2piece2hello/faibench_LLM_Infra_Bench (commit `90d7251b`).

## Φ-Bench's other 42 GPU tasks

- 39 optimize a single GPU kernel, and 3 are model-training tasks. None tests an inference server.
- On paper, 40 of the 42 would run on the 5090 after environment changes and a recalibration.
- We did not build them. If kernel-level tasks are wanted, they are the next place to look.

## Benchmarks we reviewed but did not screen

These test GPU kernels or non-LLM systems rather than an LLM inference server:
- HPCAgent-Bench (689 HPC and ML kernels);
- MicroPerf-Bench (65 slowdowns injected into microservice systems);
- FlashInfer-Bench and SOL-ExecBench workloads (single kernels on B200).

## Where did the rest of our tasks come from?

ISO-Bench ran out of candidates that fit our GPUs. We then mined our own list of 88 performance fixes from vLLM and SGLang history, and later searched for more with the same rules. See [mined-fixes.md](mined-fixes.md).
