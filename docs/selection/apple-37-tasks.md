# Apple's 37 tasks on our GPUs

**Short answer.** Two of Apple's 37 RSI-Inference-Bench tasks are usable on our GPUs: MM runs as Apple wrote it (only hardware flags change), and ST runs after we swap in a small model and fix two verifier problems. We tried five more and none is usable. Two optimization tasks might fit on an A100, but like every optimization task they ship no reference solution. The other 28 cannot run on our hardware at all.

Our hardware: one RTX 5090 (32 GB, sm_120) and six RTX 4090s (24 GB each, sm_89, no peer-to-peer link) on two servers. See [../hardware.md](../hardware.md).

A collaborator is testing some tasks separately on A100 GPUs. Those results are not part of this repo.

## Terms used on this page

- **Pristine**: the engine tree with Apple's injected slowdown, i.e. the code an agent starts from.
- **Oracle**: Apple's reference fix, applied to the pristine tree.
- **Speedup / score**: pristine time over candidate time (or candidate throughput over pristine throughput), as Apple's verifier computes it.
- **A/A**: the pristine tree scored against itself. It shows how far a score moves when nothing changed, i.e. the noise.
- **Noise floor**: Apple's minimum score for a candidate to count as faster, 1.03 in most regression tasks.

## How do the 37 tasks split?

Apple's tasks come in three groups:
- 7 **regression** tasks: a known slowdown was put into vLLM or SGLang, and the agent must find and fix it. Each ships an oracle.
- 10 **support** tasks: make a new model run correctly. Scored on pass/fail checks.
- 20 **optimization** tasks: make a model server as fast as possible. None ships a reference solution: each `known_good.patch` is an empty placeholder.

| Outcome on our GPUs | Tasks | Count |
|---|---|---|
| Usable as is (hardware flags only) | MM | 1 |
| Usable after adaptation | ST | 1 |
| Tried, not usable | SD, AS, MX, MTP, SD3 | 5 |
| Might run on an A100, but no reference solution | `optimize_qwen3_6_27b_1xh100_open`, `optimize_llada2_mini_1xh100_sglang_open` | 2 |
| Cannot run on our GPUs | FP8, 9 support tasks, 18 optimization tasks | 28 |

We also scaled one optimization task down into a new task of our own that fits one GPU (task C, below). It is labelled as our task, not Apple's.

## Why can't our numbers be compared with Apple's leaderboard?

Apple's verifiers check the hardware before scoring, and the rule depends on the task group. We read this in the verifier code.

- **Optimization tasks (20) and one support task:** the pre-run check voids any run on a GPU that is not the declared type. That includes an A100 running an H100 task. A switch lets the run continue, but labels it "functional smoke, must not be published". The GPU-count check has no switch, so a 2-GPU task on one GPU is void whatever is set.
- **Support tasks (the other 9):** the check requires the exact GPU count, a device name containing "H100" or "B200", and 78, 79 or 170 GiB of memory per GPU. There is no switch.
- **Regression tasks (7):** there is no GPU check. But their oracle speedups and noise floors were measured on H100 or B200.
- **SD3:** there is no name check, but it needs two distinct GPUs.

So nothing we run produces a publishable Apple number. Every task we hand over carries our own calibration: the oracle measured at least 3 times, A/A at least 3 times, and deliberately broken versions to confirm that the verifier rejects them.

## All 37 tasks, one line each

Model sizes are weights only. "Can't run" means the weights or the declared GPU shape do not fit any of our machines, or a kernel the task needs exists only for H100/B200.

| # | Task | Declared hardware | Verdict on our GPUs |
|---|---|---|---|
| 1 | `optimize_dsv4_flash_4xh100_dspark_sglang_open` | 4×H100 | Can't run: DeepSeek-V4-Flash-DSpark is about 149 GiB; FlashMLA needs sm90/sm100; its CUDA 13 build needs a newer driver than our 4090 servers have. |
| 2 | `optimize_dsv4_flash_4xh100_sglang_open` | 4×H100 | Can't run: 148.7 GiB of weights; FlashMLA and FP4 experts. |
| 3 | `optimize_dsv4_flash_pd_8xb200_sglang_open` | 8×B200 (4 prefill + 4 decode) | Can't run: about 149 GiB per instance; B200 build; NIXL transfer. |
| 4 | `optimize_dsv4_pro_8xb200_open` | 8×B200 | Can't run: 1.6T parameters, over 800 GB even at 4 bits. |
| 5 | `optimize_glm_5_2_8xb200_mtp_open` | 8×B200 | Can't run: GLM-5.2-FP8 is about 744 GB; its attention kernels are sm90/sm100 only. |
| 6 | `optimize_glm_5_2_8xb200_open` | 8×B200 | Can't run: same model as #5. |
| 7 | `optimize_gpt_oss_120b_2xh100_open` | 2×H100, TP2 | Can't run as is (60.8 GiB plus KV cache; preflight needs 2 GPUs). Our scaled version on gpt-oss-20b gave a best configuration of only 1.015× on one 4090, and on two 4090s the traffic between GPUs (no direct link) dominates. Not usable. |
| 8 | `optimize_kimi_k2_6_8xb200_open` | 8×B200 | Can't run: 1T parameters. |
| 9 | `optimize_kimi_linear_48b_2xb200_sglang_open` | 2×B200 | Can't run: 91.5 GiB in bf16. A 4-bit community checkpoint (28.4 GiB) might fit an A100; untested, and there is no reference solution. |
| 10 | `optimize_llada2_mini_1xh100_sglang_open` | 1×H100 | Might run on an A100 (32.5 GB, too big for the 5090's usable memory). Its CUDA 13 image needs a newer driver. No reference solution, so a run can only show that the baseline starts. |
| 11 | `optimize_minimax_m2_7_4xh100_sglang_open` | 4×H100 | Can't run: about 230 GB in FP8. |
| 12 | `optimize_minimax_m3_4xb200_mtp_open` | 4×B200 | Can't run: 413 GiB; MXFP8 kernels are sm100 only. |
| 13 | `optimize_minimax_m3_4xb200_open` | 4×B200 | Can't run: same model as #12. |
| 14 | `optimize_minimax_m3_pd_8xb200_sglang_open` | 8×B200 | Can't run: same model as #12, plus NIXL. |
| 15 | `optimize_qwen3_5_397b_a17b_fp8_8xh100_mtp_sglang_open` | 8×H100 | Can't run: about 400 GB. A same-architecture 35B FP8 model (42.7 GB) might fit an A100; untested, no reference. |
| 16 | `optimize_qwen3_5_397b_a17b_fp8_8xh100_open` | 8×H100 | Can't run: same as #15. |
| 17 | `optimize_qwen3_6_27b_1xh100_open` | 1×H100 | Might run on an A100 (55.6 GB). No reference solution. We scaled it down to Qwen3.5-4B on one GPU as **task C**, which runs (below). |
| 18 | `optimize_qwen3_6_27b_pd_2xh100_open` | 2×H100, prefill/decode split | Can't run: 55.6 GB on each of 2 GPUs, NIXL, no peer-to-peer link on our cards. |
| 19 | `optimize_qwen3_6_27b_pd_bulk_2xb200_sglang_open` | 2×B200 | Can't run: as #18, plus a B200 build. |
| 20 | `optimize_qwen3_8_27b_1xb200_dflash2_open` | 1×B200 | Can't run: about 113 GiB with the drafter, KV cache and state. |
| 21 | **AS** `regression_async_spec_decode_throughput` | 1×B200 | Tried twice, not usable: the defect is real on our cards, but Apple's fix gains under 1% (B200: 1.17×). |
| 22 | **FP8** `regression_fp8_decode_throughput` | 8×B200 | Can't run: an NVFP4 DeepSeek-R1 of about 400 GB; needs an FP8-query MLA backend (sm90/sm100). |
| 23 | **MTP** `regression_mtp_decode_throughput` | 4×H100 | Tried, not usable: the pinned vLLM supports only an 80B model; our adapted version gains about 1% inside ±5% noise. |
| 24 | **MM** `regression_multimodal_prefill_ttft` | 1×H100 | **Usable as is.** Oracle 1.55× on the 5090 and 1.39× on the 4090 (H100: 1.62×). Its verifier can be gamed. |
| 25 | **MX** `regression_mxfp4_moe_decode_throughput` | 1×H100 | Tried, not usable: the oracle switches on only for H100-class GPUs, so on ours it changes nothing. |
| 26 | **ST** `regression_sglang_stream_decode_tps_c1` | 4×B200, TP4 | **Usable after adaptation** (Qwen2.5-1.5B on one GPU, two verifier edits). Oracle 1.14× on the 4090, 1.19-1.22× on the 5090 (B200: 1.40×). |
| 27 | **SD** `regression_spec_decode_acceptance_drop` | 1×H100 | Tried, not usable: the pristine tree crashes at the largest load, and Apple's own fix fails Apple's exact-output check on our GPUs. |
| 28 | `support_deepseek_v41_flash_4xb200_vllm` | 4×B200 | Can't run: 475 GiB; sm100-only build; preflight needs B200s with ≥170 GiB. |
| 29 | `support_deepseek_v4_flash_4xb200_sglang` | 4×B200 | Can't run: 148.7 GiB; FlashMLA; preflight as #28. |
| 30 | `support_diffusiongemma_26b_a4b_2xh100_vllm` | 2×H100 | Can't run: the task is Hopper paged attention in CuTe-DSL; preflight needs H100s with ≥79 GiB. |
| 31 | `support_glm53_flash_4xb200_sglang` | 4×B200 | Can't run: 305.8 GiB; preflight as #28. |
| 32 | `support_gpt_oss_120b_4xh100_sglang` | 4×H100 | Can't run: Hopper MXFP4 path; preflight needs 4 H100s with ≥79 GiB. |
| 33 | `support_inkling_small_4xb200_sglang` | 4×B200 | Can't run: 159 GiB in NVFP4; SM100 attention. |
| 34 | `support_kimi_linear_48b_4xh100_vllm` | H100s (the config asks for 8) | Can't run: 91.5 GiB; preflight needs H100s with ≥78 GiB. |
| 35 | `support_minimax_m3_4xb200_vllm` | 4×B200 | Can't run: 413.3 GiB; MXFP8 sm100. |
| 36 | `support_qwen38_flash_next_4xb200_vllm` | 4×B200 | Can't run: 172.8 GiB; TensorRT-LLM sm100 kernels. |
| 37 | **SD3** `support_sd3_2xh100_sglang` | 2×H100 | Tried, not usable: 9 of its 49 required checks need 2 GPUs, and its full-precision reference needs about 29 GB per GPU. |

## Which kinds of regression carry over to other GPUs?

Across the regression tasks we tried, a pattern shows:
- **Mechanisms that are about the host or the scheduler carry over:** a CPU waiting on the GPU (ST), a scheduler handing the runner uneven batches (AS), kernels recompiled per shape (MM, MTP).
- **Mechanisms tied to one backend or one kernel configuration do not:** MX's fix is gated to sm90; SD's fix assumes an attention backend our cards don't select.
- **Even when the mechanism carries over, the score often does not.** Of the regressions we ran, only MM's and ST's oracles clear the 1.03 floor on our cards. AS reproduces its defect perfectly and still scores 1.00.

## What did each task we tried show?

### MM: `regression_multimodal_prefill_ttft` (usable as is)

**Symptom.** SGLang takes too long to produce the first token for image prompts, because a Triton kernel is recompiled for every new prompt length. Model Qwen3.5-4B.

**What we changed (both arms):**
- `--attention-backend triton`: SGLang's default FlashInfer fails its own version check off Hopper/SM100.
- `--mem-fraction-static` 0.50 on the 4090 and 0.60 on the 5090 (Apple: 0.85; higher values ran out of memory in the output check).
- The GPU "reap" step in `test.sh` removed: it kills every process on the machine's GPUs.
- Fixed seeds; a per-run lock and timeout; harness edits only for these. No engine or scoring change.

**Numbers.**

| GPU | Oracle | A/A | Time per verifier pass |
|---|---|---|---|
| RTX 4090 | 1.373 / 1.386 / 1.394 / 1.401 (mean 1.389, 3 cards) | 0.990-1.021 (5 runs) | about 20-27 min |
| RTX 5090 | 1.549 / 1.560 (independent tester: 1.542) | 0.997 (tester: 1.003) | about 18 min |
| H100 (Apple) | 1.620 | - | - |

**Caveats.**
- **The verifier can be gamed.** A candidate that does not fix the bug but pre-compiles the scored kernel variants at start-up earns up to 64% of the oracle's reward, and passes every check. Even without the seed, pre-compiling every possible shape for one image earns about 46%. Details in [../findings.md](../findings.md).
- **Storage matters most.** With the compile cache and temporary files on network storage, the 4090 score rises to 1.65-1.67, because the pristine arm pays extra file I/O per compile. Keep `$TMPDIR` on local disk.
- **The floor is thin on the 4090.** 1.03 is only about 2.3 A/A standard deviations above the A/A mean, so a no-op edit passes about 1% of the time. We suggest a floor of 1.05 there.
- Memory headroom is about 1 GB at fraction 0.50; do not raise it.

### ST: `regression_sglang_stream_decode_tps_c1` (usable after adaptation)

**Symptom.** SGLang streams tokens slowly for a single user, because of a GPU-to-CPU sync on every decoded token. Apple uses gpt-oss-120b on 4 B200s.

**What we changed:**
- **Model:** Qwen2.5-1.5B-Instruct on one GPU instead of gpt-oss-120b on four. gpt-oss-20b loads on the 4090 but decodes at 42.5 ms per token; the defect adds a fixed ~0.7 ms, so its ceiling is 1.017×, below the floor. On the 1.5B model the same defect costs about 15% of decode time.
- **Verifier edit 4:** one correctness probe fed random token ids. A 1.5B model loops on those, so even the oracle failed (`correctness_failed`, three runs). We use a fixed English paragraph instead; the probe still catches the broken variant it exists for.
- **Verifier edit 5:** the benchmark client counted output tokens by re-tokenizing streamed text. About 3 in 10,000 requests were badly undercounted, which moved a run's score by about 5% (one independent A/A read 0.975). We use the server's token count instead.
- `tp_size` 1, `--mem-fraction-static` 0.80 (0.60 on the 5090), GPU reap removed.

**Numbers.**

| GPU | Oracle | A/A | Time per verifier pass |
|---|---|---|---|
| RTX 4090 | 1.142-1.146 (independent tester: 1.141) | 0.998-1.003 | 18-20 min |
| RTX 5090, Triton attention + PyTorch sampling | 1.191 / 1.194 / 1.195 | 1.004 / 0.996 | about 16-17 min |
| RTX 5090, FlashInfer with a CUDA 12.9 compiler | 1.224 / 1.224 / 1.222 | 0.998 / 0.992 | about 16-20 min |
| 4×B200 (Apple, original model) | 1.398 | - | - |

Both broken variants Apple ships are caught on both GPUs. One of them (the minimum-token penalty removed) is caught only by the floor probe, not by the output digest. The two 5090 variants give different scores, so a 5090 number must name its variant. See [../hardware.md](../hardware.md).

### SD: `regression_spec_decode_acceptance_drop` (not usable)

**Symptom.** Speculative decoding (Llama-3.1-8B with an EAGLE3 drafter) accepts too few draft tokens. Apple's oracle gives 1.156× on H100.

**What happened.**
- With Apple's fp8 KV cache, our cards select the FlashInfer attention backend; H100 selects FlashAttention. Apple's oracle drops half of the upstream fix on the premise that the backend never reads a CPU-side copy of the sequence lengths. On FlashInfer it does.
- **The pristine tree crashes at the largest load** (32 concurrent requests): 3 of 3 runs on the 5090 (two illegal-address errors, one hang), 1 of 1 on the 4090. The verifier then has no denominator and fails closed.
- **On the 5090 the oracle is slower** at the loads that do run (0.899 and 0.968, geomean 0.933), and fails the first-token latency guard (0.85 against 0.95). A profiler shows why: the oracle adds one blocking GPU-to-CPU copy per decode step, about 3 ms each, which eats the acceptance gain.
- **With a bf16 KV cache on the 4090** the backend becomes FlashAttention, as Apple assumes, and the crash disappears. But Apple's exact-output check then fails Apple's own fix: speculative output matches plain decoding on 4 of 8 prompts (8 needed; pristine 0 of 8).

| Card, KV type, backend | Pristine matches | Oracle matches |
|---|---|---|
| 4090, bf16, FlashAttention 2 | 0/8 | 4/8 |
| 4090, fp8, FlashInfer | 0/8 | 2/8 |
| 5090, fp8, FlashInfer | 3/8 | 3/8 |
| 5090, fp8, Triton attention | 0/8 | 2/8 |

The non-speculative reference reproduces itself 8 of 8, so the mismatch comes from speculative decoding (an 8-token verify step does not round like a 1-token decode step), not from run-to-run noise. Making this task portable would mean replacing its exact-output check with a tolerance check, which changes the task.

### AS: `regression_async_spec_decode_throughput` (not usable)

**Symptom.** With async scheduling and speculative decoding, the scheduler hands the model runner uneven ("ragged") batches, which cannot replay the full CUDA graph. Apple's oracle gives 1.17× on B200.

**Two adapted runs, both null:**
- **5090:** gpt-oss-20b with an EAGLE3 drafter, Triton attention, concurrency divided by 4. The oracle removes the defect completely (ragged decode steps 102 → 0 in the probe; 193 → 0 at full size; outputs bit-identical). Throughput does not move: oracle 1.0002 against A/A 1.0032. Forcing every step onto the slower graph path costs only 3.2% at 32 concurrent requests and nothing at 384, and only 10-14% of steps are ragged.
- **4090:** Qwen3-1.7B with an EAGLE3 drafter on FlashAttention 2, chosen because that backend, like B200's, builds full graphs only for uniform batches. We first measured the cost of a ragged step directly (full graphs against forced piecewise graphs, on the pristine tree). Full graphs were not faster: 0.960 at 64 requests, 0.994 at 128. We had predicted 1.02-1.06. So no fix can reach 1.03, and we stopped there as planned.

On two backends and two consumer cards the defect is real and costs nothing measurable. The B200 gain must come from B200-side conditions we cannot test.

### MX: `regression_mxfp4_moe_decode_throughput` (not usable)

**Symptom.** gpt-oss-20b MoE decoding picks a bad split-k launch configuration. Apple's oracle gives 1.426× on H100.

- **On the 5090** SGLang picks a different MoE backend (FlashInfer MXFP4, because its Blackwell check covers compute capability 12), so the code with the defect never runs. We stopped before benchmarking, as planned.
- **The oracle is a no-op off H100.** It only adds `split_k = 1` when the GPU is sm90. On the 4090 and A100 the buggy path does run, but the fix is gated off, so pristine and oracle are the same code (read from the source; not run).
- Making it work would need a new oracle gated to our GPUs, i.e. a new task.

### MTP: `regression_mtp_decode_throughput` (not usable)

**Symptom.** A GDN decode kernel is recompiled for every batch shape when multi-token-prediction speculation is on.

- **The original cannot run:** the pinned vLLM knows only one model with this kernel and an MTP head, Qwen3-Next-80B (about 160 GB in bf16, 45 GB even at 4 bits). The small test checkpoints of that architecture have random weights and no MTP head.
- **Adapted version on a 4090:** vLLM 0.19.1 with Qwen3.5-4B, with Apple's defect re-injected into the successor kernel that Qwen3.5 actually calls (Apple's two lines do not compile as is in Triton 3.6; a third line was needed).
  - The defect is exercised: the pristine compiles 31-32 kernel variants per scoring, the oracle 1.
  - But each variant compiles once and is then cached, so the cost lands only on the first workload. Full scores: 1.009 (and a voided 0.982), against an A/A of 0.995. Acceptance at temperature 1 varies by ±5% per workload from run to run.
  - The correctness probe was unreliable: 2 of 11 compile caches produced a broken log-probability probe, one in each arm type.

### SD3: `support_sd3_2xh100_sglang` (not usable)

- **It needs two GPUs.** Binary reward needs all 49 checks to pass, and 9 of them start the engine with 2 GPUs (tensor, sequence and CFG parallelism). On one 5090 the default schedule exits at once.
- **Its reference generation does not fit 24 or 32 GB.** The full-precision reference loads the whole pipeline onto one GPU: 28.87 GiB of weights in fp32, 23.46 GiB for the mixed set. Any failed reference set ends the verifier.
- **On two 4090s** the pinned SGLang installs and passes a smoke test, and the two cards can talk (no peer-to-peer link; NCCL over shared memory). But the model is gated and those servers have no Hugging Face login, and the fp32 and mixed references still exceed one card.
- A serial schedule with only the bf16 references would allow at most 32 of 49 checks, which is no longer this task.

## What about the optimization tasks?

All 20 ship an empty `known_good.patch`, so a run on any hardware can show that the baseline starts but not that a better answer exists. Two might run on an A100 (#10 and #17); a collaborator is testing on A100s separately.

We scaled #17 down into a task with a reference answer of our own, **task C** (`optimize_vllm_qwen35_4b_high_concurrency`):
- Qwen3.5-4B (same architecture family as Apple's Qwen3.6-27B) on one 4090 or 5090, with Apple's method: hidden workloads, fresh random prompts per run, a geometric-mean throughput score, and Apple's output check.
- Heavily concurrent workloads, where this model leaves room to improve.
- The reference answer changes two vLLM settings and scores about 1.32× on both GPUs. It is partly tuned to the workload shapes: on unseen shapes it keeps about a third of its gain on the 4090 and under a quarter on the 5090.
- An earlier version with Apple's original three workloads (task B) gave the reference only 4%, too little to use.
- An independent tester reproduced task C on both GPUs. See [../acceptance/batch1.md](../acceptance/batch1.md) and [../acceptance/batch1-5090.md](../acceptance/batch1-5090.md).
