# Reference solution: Apple's `min_new_tokens_no_host_sync.patch`

The oracle is Apple's own task-local patch, `solution/min_new_tokens_no_host_sync.patch`, applied by Apple's
`solution/solve.sh` (task `tasks/regression_sglang_stream_decode_tps_c1`, Apple RSI-Inference-Bench commit `1a38072f`).
We do not copy it (Apple Sample Code License). `environment/scripts/stage_ws.sh <ws> fix` runs Apple's `solve.sh`
from the Apple checkout that `fetch_base.sh` makes.

- Upstream: the repair is named in the description of SGLang PR #28363
  (https://github.com/sgl-project/sglang/pull/28363), "makes the min_new_tokens penalizer non-synchronizing", but that
  hunk is absent from the PR's merged files. Upstream main later carries the same idea. The regression itself came
  with PR #26380; the pinned tree `163b9701` is that PR's merge commit.
- What it changes, per Apple's task.toml: the min-new-tokens penalizer stops indexing logits with a boolean mask (which
  forces a device-to-host copy of the selected-element count every decode step) and uses an in-place masked fill. It
  also adds a penalizer flag that lets the batch skip a per-step copy when no active penalizer needs the output ids.
  Apple states the change is output-preserving by construction (bitwise-identical logits in float32 and bfloat16).
- Apple names two other legitimate routes: making the per-step host-to-device copy asynchronous, and restoring the
  overlap that PR #26380 removed. The score does not care which route a candidate takes.

With the oracle the fixed tree scores 1.142-1.146 on the RTX 4090 and 1.19-1.22 on the RTX 5090 (README.md, section 3).
