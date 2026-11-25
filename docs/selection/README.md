# How the tasks were chosen

A usable task needs four things:
- a real slowdown;
- a known correct fix;
- a speedup from that fix that sits well above run-to-run noise on our GPUs;
- an output check that rejects broken versions of the fix.

Most candidates fail at least one of these. This table shows how many survived from each source.

| Source | Considered | Measured on GPU | Tasks | Details |
|---|---|---|---|---|
| Apple RSI-Inference-Bench | 37 tasks | 7 tried | 2 (one as is, one with a smaller model) | [apple-37-tasks.md](apple-37-tasks.md) |
| ISO-Bench (merged vLLM/SGLang performance fixes) | its vLLM/SGLang fixes | 29 | 7 (8 passed; two are the same change merged twice) | [iso-bench-screen.md](iso-bench-screen.md) |
| Fixes we mined from vLLM/SGLang history | 88 listed, plus 9 found later | 22 (2 turned out not runnable) | 11 | [mined-fixes.md](mined-fixes.md), [mined-candidates.csv](mined-candidates.csv) |
| Φ-Bench | 43 GPU tasks | 1 ported | 1 | [other-benchmarks.md](other-benchmarks.md) |
| Our own, Apple-style | an Apple optimization task scaled down to one GPU | 1 | 1 | [../../tasks/optimize_vllm_qwen35_4b_high_concurrency/](../../tasks/optimize_vllm_qwen35_4b_high_concurrency/) |
| **Total** | | | **22** | |

**What decided it:**
- **Apple's tasks** mostly need data-center GPUs: models of hundreds of GB, multi-GPU links, or kernels that run only on H100/B200. Of the few that fit, some don't show their slowdown on consumer cards.
- **Upstream fixes** passed when two things held:
  - the fix removed work on every step, such as GPU-to-CPU syncs, kernel launches or a repeated CPU scan;
  - the load was bound by the GPU, or by CPU work that runs between GPU steps.

  Removing a sync gave nothing on a CPU-bound load, where the sync was not on the critical path. "Enable feature X" PRs have no working baseline to compare against.
- **The load mattered as much as the fix.** Two real fixes showed no gain on the load we had fixed in advance, but did on a load designed for them.
- **Writing a prediction down before each run** helped. For every mined candidate, we recorded the expected speedup and our confidence before building anything. That record is how we measured our hit rate honestly.

More batches of mined fixes are being screened. A task is added to this repo only after its package has been rebuilt from an empty directory and has reproduced its numbers. Independent rebuilds follow; [../acceptance/](../acceptance/) shows where each task stands.
