# Tasks

There are 21 tasks, one directory each. Every task is a real slowdown in vLLM or SGLang with a known reference fix. You give an agent the slow code, and score its edited tree against the unmodified code on the same GPU. Each directory follows Apple's RSI-Inference-Bench layout.

## What is in a task directory?

| File | What it is | Give it to the agent under test? |
|---|---|---|
| `instruction.md` | The task as an agent sees it: the symptom, how it is scored, how to measure, the rules | **yes** |
| `task.toml` | Metadata: upstream repo and commits, model, metric, reference speedup, noise band, gate | no: it names the fix commit |
| `README.md` | The card for humans: load, reference result, noise, output checks, rejected cheats, limits, cost | no |
| `solution/` | The reference fix: upstream PR, files changed, how the package builds it | no |
| `results/summary.json` | Every calibration and scoring job, variant verdicts, acceptance numbers | no |
| `package/` | Self-contained scoring package for the `fix_*` tasks: pinned environments, scorer, loads, setup scripts. Start at `package/RUN.md` | no: it builds the reference fix and broken variants of it |
| `environment/`, `tests/` | For the four tasks without a package: setup scripts, Dockerfile diffs and the scorer, in Apple's naming | `environment/` only where the README says so |

**Keep the answer away from the agent.** Give it `instruction.md` and a copy of the unmodified source tree only. Run the package or scorer outside its reach, then score its tree with them.

## How do I run one?

- **`fix_*` tasks (17):** follow `package/RUN.md`. Setup takes 10-50 minutes and needs internet; scoring runs offline. Calibrate once per GPU with `harness/run.sh <label> cal`, then score each candidate with `harness/run.sh <label> score <name>`. Each `instruction.md` says how a candidate tree is supplied.
- **Apple tasks, task C, Φ-Bench (4):** follow the "How to run it" section of the task's `README.md`. These use Apple's or Φ-Bench's own verifier, with our changes shipped as diffs and scripts.

Read [../docs/hardware.md](../docs/hardware.md) first: it covers CPU pinning, `TZ=UTC` and what the RTX 5090 needs.

## Which task should I start with?

- **Quickest to score:** #6994 (about 4.5 min per scoring), #7822 (4.7 min), #14956 (7.5 min).
- **Largest margin over noise:** #16521, at 3.83× with 0.6% noise.
- **Runs on both GPU types:** the Apple tasks, task C, Φ-Bench, #16521, #20972, #37338 and #27868. On the 5090, #14956 is measured; #35442, #27418 and #30056 are still being measured.
- **Weakest:** #20503, an 8% effect against noise of up to 3%. #27868 and #35442 have small gains (3-4%) with small noise.
- **Related tasks:**
  - #9460 and #14956 fix the same code path; count them as one family.
  - `fix_vllm_14857_mamba2_prefill` also ships `package-14778/`, a second merge of the same change.

## All tasks

The speedups are on an RTX 4090, the median over scoring jobs. Minutes are per scoring job on the 4090. "5090" says whether the task runs on an RTX 5090.

### From Apple's RSI-Inference-Bench, and Apple-style

| Task | Engine | Model | Speedup | Min | 5090 | Independent rebuild |
|---|---|---|---|---|---|---|
| [apple_regression_multimodal_prefill_ttft](apple_regression_multimodal_prefill_ttft/) | SGLang | Qwen3.5-4B | 1.39× | 27 | yes, 1.55× | PASS-with-fixes |
| [apple_regression_sglang_stream_decode_tps_c1_small](apple_regression_sglang_stream_decode_tps_c1_small/) | SGLang | Qwen2.5-1.5B-Instruct | 1.14× | 20 | yes, 1.22× | PASS-with-fixes (4090) |
| [optimize_vllm_qwen35_4b_high_concurrency](optimize_vllm_qwen35_4b_high_concurrency/) | vLLM | Qwen3.5-4B | 1.31× | 19 | yes, 1.32× | PASS-with-fixes |
| [phibench_vllm_scheduler](phibench_vllm_scheduler/) | vLLM | Qwen2.5-1.5B-Instruct | 1.65× | 24 | yes, 1.61× | setup works; score 3-4% lower on a quieter host |

### From ISO-Bench's list of upstream fixes

| Task | Engine | Model | Speedup | Min | 5090 | Independent rebuild |
|---|---|---|---|---|---|---|
| [fix_vllm_7822_prefix_caching](fix_vllm_7822_prefix_caching/) | vLLM | Llama-3-8B-Instruct-FP8 | 1.48× | 4.7 | no | not yet |
| [fix_vllm_12380_cuda_graph_decode](fix_vllm_12380_cuda_graph_decode/) | vLLM | Llama-3-8B-Instruct-FP8 | 1.51× | 8.7 | no | PASS-with-fixes |
| [fix_sglang_6994_lora_host_sync](fix_sglang_6994_lora_host_sync/) | SGLang | Qwen2.5-1.5B-Instruct | 1.47× | 4.5 | no | PASS-with-fixes |
| [fix_vllm_16436_sampler_host_sync](fix_vllm_16436_sampler_host_sync/) | vLLM | Qwen2.5-1.5B-Instruct | 1.09× | 19 | no | not yet |
| [fix_vllm_13365_ngram_draft_lookup](fix_vllm_13365_ngram_draft_lookup/) | vLLM | Qwen2.5-1.5B-Instruct | 1.09× | 9.6 | no | PASS-with-fixes |
| [fix_vllm_14930_spec_decode_verify](fix_vllm_14930_spec_decode_verify/) | vLLM | Qwen2.5-1.5B-Instruct | 1.09× | 12 | no | PASS-with-fixes |
| [fix_vllm_14857_mamba2_prefill](fix_vllm_14857_mamba2_prefill/) | vLLM | Bamba-9B | 1.06× | 9.5 | no | PASS-with-fixes |

### Mined by us from vLLM and SGLang history

| Task | Engine | Model | Speedup | Min | 5090 | Independent rebuild |
|---|---|---|---|---|---|---|
| [fix_sglang_16521_multiturn_prefix_reuse](fix_sglang_16521_multiturn_prefix_reuse/) | SGLang | Qwen2.5-1.5B-Instruct | 3.83× | 9 | yes, 3.52× | PASS |
| [fix_sglang_20972_logprob_host_sync](fix_sglang_20972_logprob_host_sync/) | SGLang | Qwen2.5-1.5B-Instruct | 1.14× | 10 | yes, 1.23× | PASS |
| [fix_sglang_9460_mrope_decode_sync](fix_sglang_9460_mrope_decode_sync/) | SGLang | Qwen2-VL-2B-Instruct | 1.20× | 9 | no | PASS |
| [fix_vllm_37338_cold_start_autotune](fix_vllm_37338_cold_start_autotune/) | vLLM | Qwen3.5-4B | 1.29× | 27 | yes, 1.18× | PASS |
| [fix_sglang_27868_dual_stream_decode](fix_sglang_27868_dual_stream_decode/) | SGLang | Qwen3.5-4B | 1.03× | 11 | yes, 1.06× | PASS |
| [fix_vllm_35442_spec_accept_count_copy](fix_vllm_35442_spec_accept_count_copy/) | vLLM | Qwen3.5-4B | 1.04× | 16 | being measured | PASS-with-fixes |
| [fix_vllm_27418_vit_patch_embed](fix_vllm_27418_vit_patch_embed/) | vLLM | Qwen2-VL-2B-Instruct | 1.69× | 11 | being measured | PASS-with-fixes |
| [fix_sglang_14956_mrope_text_sync](fix_sglang_14956_mrope_text_sync/) | SGLang | Qwen2-VL-2B-Instruct | 1.31× | 7.5 | yes, 1.49× | PASS |
| [fix_vllm_30056_reasoning_end_scan](fix_vllm_30056_reasoning_end_scan/) | vLLM | Qwen3-0.6B | 1.79× | 8.6 | being measured | in progress |
| [fix_sglang_20503_waiting_queue_set](fix_sglang_20503_waiting_queue_set/) | SGLang | Qwen2.5-0.5B-Instruct | 1.08× | 10 | not tried | in progress |

"PASS-with-fixes" means an independent tester reproduced the numbers but had to fix the instructions or the harness. Those fixes are now in the package. "Not yet" means only the authors have rebuilt the package from an empty directory. See [../docs/acceptance/](../docs/acceptance/).
