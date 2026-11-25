# iso13365 prediction (written 2026-11-24 08:2x UTC, before any calibration or smoke run; not edited afterwards)

Inputs: the screen (`SCREEN_BATCH2.md`: steady-state reps 0.72 vs 0.76 s, AB 0.947–0.949; A/A 0.998–1.003), the PR
diff, and the parent's `gpu_model_runner.generate_draft_token_ids` / `NgramProposer` source. No new measurement.

Official load (my harness, not the screen's): Qwen2.5-1.5B, V1, n-gram k = 5, n = 3 (min; the code ignores max),
32 × 2048-token repeated-40-token-segment prompts, 128 greedy tokens. Per rep a new prompt set, prefilled untimed
(max_tokens = 1), then the timed decode. Metric = sum of 6 timed reps.

Mechanism I expect: the parent's KMP loop is Python over numpy scalars. It returns at the FIRST match of the last
3 tokens, and with a repeated segment that match lies inside the first ~43 tokens of the prompt. So the parent's cost
per call is a fixed ~20–60 µs (call overhead + a short scan), not proportional to the context. 32 calls per step.

| Quantity | Prediction |
|---|---|
| REF (parent time / fix time), calibration, 5 pairs | **1.055** (plausible 1.045–1.065) |
| Between-job SD of REF (≥ 4 jobs, 2 hosts) | **0.4 %** relative (plausible 0.2–0.8 %); CPU-bound gaps react to host load |
| A/A max deviation, 5 pairs | ≤ 0.6 % |
| n1 vs main | same REF within 1 % if the CCD is quiet; n1's host is shared with another user, so maybe 1–2 % lower |
| Outputs | bit-identical parent vs fix; proposals, acceptances and step counts identical |
| Accepted tokens per verify step (repeat load) | ~4.5–5 of 5 (the model copies the repeated segment) |
| Transfer: random prompts (no n-gram match anywhere) | parent scans the whole 2048+ context in Python every call: **3–6×** |
| Transfer: late match (1024 random + repeated tail) | scan ~1024 tokens per call: **1.3–2×** |
| Transfer: long (IN 3968, repeat) | same early match, so the same absolute saving per step while the GPU step is longer: **1.04–1.05, i.e. NOT larger** than the official load (contradicts the screen note "grows with context length × batch") |
| Transfer: batch 64 (N = 64, IN 2048, repeat) | CPU saving per step doubles, GPU step grows < 2×: **1.07–1.09** |
| Cost per scoring job | ~9 runs × ~45 s ≈ 7 min GPU time (+ ~2 min integrity hashing on CPU) |
| Peak VRAM | ~20 GB (0.80 utilisation; screen peak 20.0 GB) |
| Canaries | "propose nothing" and "k = 3" caught by the counters; a search capped to the last W tokens gives the SAME proposals on a periodic load, so on the official load it is caught only if the cap changes which match is found — I expect the correctness run's unit test and its random / late groups to catch it, not the timed check |
