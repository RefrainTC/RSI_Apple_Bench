# Reference solution: SGLang PR #31558

The oracle is Apple's own: `solution/oracle_fix.patch` in Apple's task directory
(`tasks/regression_multimodal_prefill_ttft`, Apple RSI-Inference-Bench commit `1a38072f`). It is the merge-commit
diff of SGLang PR #31558, "[Performance] Avoid FLA L2-norm recompilation by token count"
(https://github.com/sgl-project/sglang/pull/31558), squash merge `42a058c760c768ce37739109de6a430cce49cc51`, whose
single parent is the task's pinned tree `01a96720`.

- File changed: `python/sglang/kernels/ops/attention/fla/l2norm.py` (1 file, +2/-5).
- What it does: the L2-norm kernel stops treating the flattened token count as a compile-time constant, and an unused
  constant derived from it is deleted. One cubin then serves every prompt length, instead of one per length.
  Numerics are identical (16,965 of 16,965 logprob positions matched, max delta 0, on both GPUs).

We do not copy Apple's patch (Apple Sample Code License). `environment/scripts/stage_ws.sh <ws> fix` applies it from
the Apple checkout that `fetch_base.sh` makes, with `patch -p2`, exactly as Apple's `solve.sh` does, and checks that
the fix's term is present afterwards.

The fixed arm then scores 1.37-1.40 on the RTX 4090 and 1.54-1.56 on the RTX 5090 (see README.md, section 3).
