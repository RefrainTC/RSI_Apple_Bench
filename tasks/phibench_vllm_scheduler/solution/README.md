# Reference solutions: config-only launch scripts

Phi-Bench's task ships its own oracle, `solution/oracle_v2_launch_server.sh` (in the task, applied by
`solution/solve.sh --oracle`). It launches stock vLLM with `--max-num-seqs 128 --async-scheduling
--api-server-count 4` over the baseline's flags. Two facts matter:

- **`--api-server-count 4` has no effect in it.** The oracle starts `python3 -m vllm.entrypoints.openai.api_server`,
  which parses that flag but never uses it; only `vllm serve` starts several API servers. The server log shows one
  API server. So the oracle's real difference from the baseline is `--max-num-seqs 64 → 128` plus
  `--async-scheduling`.
- **It is not the strongest config-only launch on our GPUs.** We reran the authors' 13 sweep variants plus a
  `max-num-seqs × --async-scheduling` grid (`environment/scripts/make_variants.py`):

| GPU | Strongest config-only launch | Score (3 scorings, median) | Task oracle |
|---|---|---|---|
| RTX 5090 | `variants/x_s192_a_serve_api4.sh`: max-num-seqs 192, async, launched with `vllm serve` and 4 API servers | 1.6088 (1.6009 / 1.6120 / 1.6088) | 1.3633 |
| RTX 5090, task's own entry point | `variants/g_s192_a.sh`: max-num-seqs 192, async (ties with 256 and 384) | 1.5165 | |
| RTX 4090, task's own entry point | max-num-seqs 384 + async (`g_s384_a`; 192 and 256 tie within 0.1%) | 1.6477 | 1.3927 |

The gain comes from two independent switches. `max-num-seqs >= 192` helps only the three bursts wider than 128
(1.31-1.39x over the oracle on the 4090, 1.11-1.28x on the 5090). `--async-scheduling` helps every burst a little
(+7-9% on the 4090). `vllm serve` with 4 API servers was not run on the 4090, so 1.6477 is the strongest launch with the
task's own entry point there, not necessarily the strongest overall. Config rankings differ between the two GPUs
(Spearman 0.69 over 14 valid candidates), so the sweep must be redone per GPU.

We use the strongest config-only launch as REF: launch flags are a legal submission, and an agent can find them. With
the oracle-only REF, config-only submissions earned reward 0.25-0.51 on the 4090.
