# Prediction for #12380 calibration (written 2026-11-24 07:55 UTC, before any run of this task; not edited later)

Inputs: the batch-2 screen only (GPU1, shared CCD during venv builds and other screens; fix 1.44-1.46 s per rep in
30/30 reps, parent >= 2.17 s, parent A/A up to 27 %), and the diff (`decode_only` reset per batch; in the parent the
persistent builder keeps `decode_only=False` after the profile-run prefill, so every decode step runs eager).

Task load (fixed before any run): Llama-3-8B-Instruct-FP8, CUDA graphs on (defaults), batch 8, 32 random-token prompt
ids, 128 greedy output tokens (ignore_eos), 10 timed reps with a different prompt set per rep; metric = mean rep
latency. Pinned to one CCD that no other job uses.

| Quantity | Prediction |
|---|---|
| Fix rep latency | 1.44-1.47 s (graph replay, GPU-bound, ~11 ms per decode step) |
| Parent rep latency | 2.15-2.25 s on an exclusive CCD (CPU launch-bound, ~17 ms per decode step) |
| REF (calibration, 5 pairs) | **1.50** (plausible range 1.45-1.56) |
| A/A max deviation (5 pairs, exclusive CCD) | 1.0 % (could reach 2-3 % if host noise hits the parent) |
| Between-job SD of REF (>= 4 jobs) | **~1.0 % of REF** (higher than #7822's 0.17 % and #16436's 0.54 %, because the parent is CPU-bound and follows host state); between-GPU/host offset up to 2 % |
| Transfer: batch 1 | ~1.7x (same CPU launch cost per step, less GPU work per step) |
| Transfer: batch 32 | ~1.35x (more GPU work per step, launch cost roughly the same) |
| Transfer: 512 output tokens (batch 8) | ~1.5x (prefill share smaller, attention grows a little) |
| Output differences fix vs parent | Graph replay at batch 8 pads to 8 (no padding), so token ids identical or only rare near-tie divergences |
| Cost per scoring job | ~12 min of GPU time (correctness run + 2 warm-ups + 3 pairs, ~75 s per run) |
| Peak VRAM | ~20.1 GB (0.82 of 24 GB), as in the screen |

Risks named in advance: the parent's CPU-bound speed makes REF a property of the host CPU as much as of the GPU, so the
n1 host (same 5975WX, but shared with another user's job) may give a different REF; the in-process step-trace and
graph counters can in principle be bypassed by candidate code in the same process.
