# RSI_Apple_Bench: inference-optimization tasks that run on consumer GPUs

Apple's [RSI-Inference-Bench](https://github.com/apple-aiml-research/ml-rsi-inference-bench) asks coding agents to speed up real LLM inference engines (vLLM, SGLang) on data-center GPUs: H100s and B200s, with models of up to hundreds of GB. We wanted to know how much of that can be tested on consumer cards: one RTX 5090 (32 GB) and RTX 4090s (24 GB).

The answer: very little of Apple's benchmark runs as is. So we also built our own tasks, in Apple's format, mostly from real performance fixes in vLLM and SGLang history. This repo holds **22 tasks** that run and measure reliably on these GPUs. Each task has:

- a known correct answer (the reference fix);
- a speedup calibrated against run-to-run noise;
- an output check;
- a set of deliberately broken or cheating versions that the check must reject.

For 20 of the 22, an independent tester rebuilt the task from scratch and got the same numbers. For Φ-Bench the rebuild worked, but the score read 3–4% lower. The newest task (#17005) has so far been rebuilt from an empty directory only by its authors.

| | Screened | Usable here |
|---|---|---|
| Apple RSI-Inference-Bench tasks | 37 | 2: one as is, one with a smaller model |
| Fixes from ISO-Bench's list of vLLM/SGLang performance fixes | 29 measured on GPU | 7 |
| Fixes we mined from vLLM/SGLang history | 22 measured on GPU (from 88 mined + 9 found later) | 11 |
| Other | an Apple-style task scaled down to one GPU; one task ported from Φ-Bench | 2 |

## The tasks

Each GPU column gives the reference fix's speedup on the measured job (the median over several scoring jobs) and, in brackets, the noise: the largest deviation from 1.0 when the unmodified code is scored against itself (an A/A run). A task is usable only if its speedup is well above its noise. "-" means not tried on that GPU; "does not run" means the pinned builds have no RTX 5090 code; "being measured" means the 5090 runs are under way.

| Task | Origin | What is slow | RTX 5090 | RTX 4090 | Independent rebuild |
|---|---|---|---|---|---|
| [apple_regression_multimodal_prefill_ttft](tasks/apple_regression_multimodal_prefill_ttft/) | Apple, as is | SGLang: slow first token for image prompts | 1.55× (0.5%) | 1.39× (2.1%) | yes, both GPUs |
| [apple_regression_sglang_stream_decode_tps_c1_small](tasks/apple_regression_sglang_stream_decode_tps_c1_small/) | Apple, smaller model | SGLang: slow token streaming at concurrency 1 | 1.22× (0.8%) | 1.14× (0.2%) | yes, 4090 |
| [optimize_vllm_qwen35_4b_high_concurrency](tasks/optimize_vllm_qwen35_4b_high_concurrency/) | ours, Apple-style | vLLM throughput under heavy concurrent load | 1.32× (0.5%) | 1.31× (0.2%) | yes, both GPUs |
| [phibench_vllm_scheduler](tasks/phibench_vllm_scheduler/) | Φ-Bench, ported | vLLM scheduling of mixed request bursts | 1.61× (0.5%) | 1.65× (0.2%) | partly: setup works, score 3–4% lower on a quieter host |
| [fix_vllm_7822_prefix_caching](tasks/fix_vllm_7822_prefix_caching/) | vLLM history (ISO-Bench) | repeated prompt prefixes are recomputed | - | 1.48× (0.2%) | yes; the ±0.5% band is tight, so about 1 correct job in 5 lands just outside it |
| [fix_vllm_12380_cuda_graph_decode](tasks/fix_vllm_12380_cuda_graph_decode/) | vLLM history (ISO-Bench) | decode falls off CUDA graphs | - | 1.51× (2.3%) | yes |
| [fix_sglang_6994_lora_host_sync](tasks/fix_sglang_6994_lora_host_sync/) | SGLang history (ISO-Bench) | LoRA serving waits on the GPU too often | - | 1.47× (1.2%) | yes |
| [fix_vllm_16436_sampler_host_sync](tasks/fix_vllm_16436_sampler_host_sync/) | vLLM history (ISO-Bench) | the sampler waits on the GPU once per step | - | 1.09× (0.4%) | yes |
| [fix_vllm_13365_ngram_draft_lookup](tasks/fix_vllm_13365_ngram_draft_lookup/) | vLLM history (ISO-Bench) | slow n-gram draft lookup in speculative decoding | - | 1.09× (0.5%) | yes |
| [fix_vllm_14930_spec_decode_verify](tasks/fix_vllm_14930_spec_decode_verify/) | vLLM history (ISO-Bench) | slow verification step in speculative decoding | - | 1.09× (0.2%) | yes |
| [fix_vllm_14857_mamba2_prefill](tasks/fix_vllm_14857_mamba2_prefill/) | vLLM history (ISO-Bench) | slow Mamba2 prefill loop | - | 1.06× (0.3%) | yes |
| [fix_sglang_16521_multiturn_prefix_reuse](tasks/fix_sglang_16521_multiturn_prefix_reuse/) | SGLang history (mined) | turn 2 of a chat recomputes the model's previous answer | 3.52× (0.5%) | 3.83× (0.6%) | yes |
| [fix_sglang_20972_logprob_host_sync](tasks/fix_sglang_20972_logprob_host_sync/) | SGLang history (mined) | returning log-probabilities makes the CPU wait on the GPU every step | 1.23× (0.7%) | 1.14× (0.7%) | yes |
| [fix_sglang_9460_mrope_decode_sync](tasks/fix_sglang_9460_mrope_decode_sync/) | SGLang history (mined) | a vision-language model waits on the GPU per request at every decode step | does not run | 1.20× (1.9%) | yes, second server |
| [fix_vllm_37338_cold_start_autotune](tasks/fix_vllm_37338_cold_start_autotune/) | vLLM history (mined) | cold start: the first batch compiles kernels the warm-up should have compiled | 1.18× (0.6%) | 1.29× (2.5%) | yes, second server |
| [fix_sglang_27868_dual_stream_decode](tasks/fix_sglang_27868_dual_stream_decode/) | SGLang history (mined) | two small operations per layer could overlap on a second GPU stream | 1.06× (0.6%) | 1.03× (0.7%) | yes, second server |
| [fix_vllm_35442_spec_accept_count_copy](tasks/fix_vllm_35442_spec_accept_count_copy/) | vLLM history (mined) | speculative decoding: the CPU waits on the GPU every step for the accepted-token count | 1.11× (2.9%; memory setting lowered) | 1.04× (0.4%) | yes, after a harness fix |
| [fix_vllm_27418_vit_patch_embed](tasks/fix_vllm_27418_vit_patch_embed/) | vLLM history (mined) | the vision encoder's patch embedding runs a slow 3-D convolution | being measured | 1.69× (3.6%) | yes, after a harness fix |
| [fix_sglang_14956_mrope_text_sync](tasks/fix_sglang_14956_mrope_text_sync/) | SGLang history (mined) | same code path as #9460, text-only requests | 1.49× (0.7%) | 1.31× (1.4%) | yes, second server |
| [fix_vllm_30056_reasoning_end_scan](tasks/fix_vllm_30056_reasoning_end_scan/) | vLLM history (mined) | structured output with a reasoning model rescans the whole sequence every step | being measured | 1.79× (2.5%) | yes, also on a second server; a fresh calibration can fail on a rare output flip |
| [fix_sglang_20503_waiting_queue_set](tasks/fix_sglang_20503_waiting_queue_set/) | SGLang history (mined) | the scheduler rebuilds a lookup set for every waiting request | - | 1.08× (3.0%; the weakest task) | yes |
| [fix_sglang_17005_logprob_position_convert](tasks/fix_sglang_17005_logprob_position_convert/) | SGLang history (mined) | returning top-20 log-probabilities for long generations takes far longer than the GPU needs | - | 1.40× (1.8%) | not yet (authors' empty-directory rebuild only) |

A few things to keep in mind:
- #9460 and #14956 fix the same code path, so count them as one family.
- The 4090 and 5090 speedups differ in both directions. Each GPU type needs its own reference score.
- The older history tasks run only on the 4090: their pinned 2024–25 PyTorch and engine builds have no code for the 5090.

## How a task is laid out

We follow Apple's directory layout and metadata style, and add what a reproducible measurement needs. For the 11 mined `fix_*` tasks, `package/kit/test.sh` plays the role of Apple's `tests/test.sh` (see [Scoring an agent](#scoring-an-agent)). There are no container images yet. [tasks/README.md](tasks/README.md) lists every task grouped by origin, with engine, model, minutes per scoring and GPU support:

```
tasks/<task>/
  task.toml          metadata: upstream repo and commits, model, metric, reference speedup, noise band, gate
  instruction.md     what an agent is told: the symptom, how it is scored, how to measure, the rules
  README.md          the card for humans: load, reference result, noise, output checks, rejected cheats, limits, cost
  solution/          the reference fix: upstream PR and commit, files changed, how the package builds it
  results/           summary.json: every calibration and scoring job, variant verdicts, acceptance numbers
  package/           self-contained scoring package: pinned environments, scorer, loads, setup scripts; start at RUN.md
```

**Keep the package away from an agent under test.** `package/` builds the reference fix as its own arm, and `package/variants/` holds broken versions of that fix. `package/RUN.md` also names the files the fix changes. Give an agent `instruction.md` and an agent view only. For the 11 mined tasks, `package/kit/make_agent_view.sh` builds that view. Then score its tree yourself.

The packages are copies of the ones we verified. In the public copies, internal host names and home-directory paths are replaced with placeholders. One Chinese-language harness README is replaced with an English translation, and five more Chinese notes are left out for now. `MANIFEST.sha256` was regenerated after those changes. Nothing else differs.

## Running a task

Each `package/RUN.md` gives the steps. For the 11 packages mined from history (and their shared harness), in short:

1. Copy `site.env.example` to `site.env` and set your GPU, CPU set, port and a local staging directory.
2. Run `export TZ=UTC`. Some setup scripts pass dates to `uv --exclude-newer`, which reads them in local time.
3. Run the setup scripts: hashed lock, model download, staging. Setup takes 10–50 minutes and needs internet.
4. Calibrate: `bash harness/run.sh <label> cal`. This scores the reference fix and an A/A run.
5. Score a candidate: `bash harness/run.sh <label> score <name>`. One scoring job takes 5–27 minutes.

The 7 ISO-Bench packages are older and use their own job scripts, which their `RUN.md` describes. The four tasks without a package use Apple's or Φ-Bench's verifier; see each task's README.

You need Linux, an NVIDIA GPU with a driver for CUDA 12.4 or newer (the 5090 needs a newer one), `uv`, and 25–65 GB of local disk per task. The GPU must be otherwise idle: the harness refuses to run if another process is using it.

## Scoring an agent

The 11 mined `fix_*` packages ship an agent kit in `package/kit/` (start at `package/kit/KIT.md`):

1. Calibrate once on your GPU type, then run `kit/set_ref.py`. This writes that GPU's reference speedup and band into `gate.json`, together with the GPU's name.
2. Run `kit/make_agent_view.sh <dir>` to build the agent's working directory. It holds a private copy of the unmodified engine, the benchmark, the scored load, the model and `dev_bench.sh`, and no trace of the fix. The script scans the result for leaks.
3. Give the agent `instruction.md` and that directory only. It edits the engine in place and times its own runs with `dev_bench.sh`.
4. Score its edited engine directory, or a diff, with `kit/test.sh`. Like Apple's `tests/test.sh`, it writes a reward file. It refuses if the GPU differs from the calibrated one. The candidate is built as a real copy, never hard-linked to the baseline. The baseline, the package and the scorer's configuration are hash-checked before, during and after the job.

We validated the kit end to end on an RTX 5090 with #16521. The untouched tree scored 1.00 (`no_gain`, reward 0), the human fix's diff scored `ties_ref` (reward 0.999), and a broken variant failed the gate. Every tamper and GPU-mismatch case was refused. See [docs/acceptance/kit-validation.md](docs/acceptance/kit-validation.md). The other packages share the same harness layout and passed static checks, but have not yet been scored through the kit on a GPU.

## What is not ready yet

- **The scorer still trusts the candidate's process.** Timing and the settings readback run inside the candidate's Python process, as the agent's own user. The kit *detects* writes to the baseline but cannot stop a candidate from faking its own clock. Run candidates in a sandbox before using scores as rewards for an adversarial agent.
- **Only the 11 mined packages have the kit.** The 7 ISO-Bench packages use an older harness. #6994 and #14857 have no candidate slot at all, and their `instruction.md` describes an untested workaround. The four tasks without a package score only the reference fix and our built-in variants.
- **NVIDIA only.** Every package pins CUDA builds of PyTorch, vLLM and SGLang and uses `nvidia-smi`. AMD GPUs (ROCm) would need the engines re-ported, which changes the task.
- **Calibration costs GPU time.** Setting REF on a new GPU type takes about 2-3 GPU-hours per `fix_*` task (one calibration and two scoring jobs).
- **No container images.** The kit runs on the host. There is no `environment/Dockerfile` for Harbor-style isolation yet, and inside a container `nvidia-smi` hides other processes, so the harness's busy-GPU guard would not work.

## What we learned

The details are in [docs/findings.md](docs/findings.md).

- **Few of Apple's tasks fit consumer GPUs.** Most need far larger models or H100/B200-only kernels. A few fit, but the problem they test doesn't show up on our cards.
- **Speedup scores are fragile.**
  - One Apple verifier can be gamed: a candidate that fixes nothing still earns up to 64% of the reward.
  - Moving a compile cache from local disk to network storage changed one score from 1.39 to 1.65.
  - A quieter host lowered another task's reference score by 3–4%.
- **A rising score is not the same as a faster system.** In a 6-round agent run, the score kept rising, but after round 3 the gains held only on the scored load.
- **Human fixes are not always correct.** One upstream fix we screened (SGLang #10709) is 1.14× faster but changes outputs far beyond rounding, so it can't serve as a reference answer.
- **Independent rebuilds catch what authors miss.** A tester found a harness bug that made one task's first calibration on a fresh install fail its own correctness check. The authors' own runs had passed by luck.
- **The load matters as much as the fix.** Two real fixes showed no gain on the load we fixed in advance. A second load, with its prediction written before the run, showed 1.69× and 1.04×.

## How we chose tasks

An index of all docs is in [docs/README.md](docs/README.md).

| Document | What it covers |
|---|---|
| [docs/selection/](docs/selection/) | How many candidates survived from each source, and what decided it |
| [docs/selection/apple-37-tasks.md](docs/selection/apple-37-tasks.md) | All 37 Apple tasks: which run on our GPUs, and why the rest don't |
| [docs/selection/other-benchmarks.md](docs/selection/other-benchmarks.md) | Other benchmarks we screened, and the Φ-Bench port |
| [docs/selection/iso-bench-screen.md](docs/selection/iso-bench-screen.md) | 29 ISO-Bench fixes measured on GPU, and what predicted success |
| [docs/selection/mined-fixes.md](docs/selection/mined-fixes.md) | The fixes we mined, screened in batches: predictions written in advance, results, hit rates |
| [docs/hardware.md](docs/hardware.md) | Our GPUs; which tasks run on the 4090 and the 5090, with numbers; setup notes |
| [docs/acceptance/](docs/acceptance/) | Reports from the independent testers who rebuilt each task from scratch |

## Licenses and sources

- The Apple-derived tasks are adapted from Apple's RSI-Inference-Bench and keep its license notices. See that repository's LICENSE.
- The `fix_*` tasks pin upstream vLLM (Apache-2.0) and SGLang (Apache-2.0) commits. The packages download them during setup and do not redistribute them.
- The Φ-Bench task is ported from Φ-Bench; its `solution/README.md` links the source.
- All measurements were made by us on our own GPUs. None of them are Apple leaderboard numbers. Apple's rules void results from other GPU types.
