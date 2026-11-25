# Acceptance batch 6: vllm30056 (N1) and sgl20503 (N2) (2026-11-25)

An independent tester, who wrote none of these packages, ran each package the way an outside user would. It copied the package alone into an empty directory on the 4-GPU 4090 server, followed `RUN.md` literally with fresh downloads of everything, and used GPU0 and GPU1. The same tester also gave #7822 and #16436 their first independent test and tried the public repo's agent path; those are in [batch6-iso.md](batch6-iso.md).

## Summary

| Package | Verdict | REF: tester vs card | A/A max deviation: tester vs card | Main finding |
|---|---|---|---|---|
| sgl20503 (SGLang #20503) | **PASS** | median of 4 jobs **1.095** (1.097, 1.0975, 1.0949, 1.0864) vs 1.08 ± 4% | 2.45% vs 2.06-2.97% | none; it reads about 1-1.5% above the authors' six jobs (1.068-1.087), inside the band |
| vllm30056 (vLLM #30056) | **PASS-with-fixes** | median of 5 jobs **1.790** (1.7899, 1.7895, 1.7909, 1.7852, 1.7962) vs 1.79 ± 3% | 2.22% (first calibration), 1.00% (second) vs 0.46% / 2.47% | the first fresh calibration ended `fail:exact` from one flipped parent rep; calibration mode has no rerun rule. A second calibration passed. |

Both packages' manifests checked clean, both builds reproduced the authors' environments exactly, every human-fix job landed in its band, and every broken variant got the verdict its card lists. GPU time: about 3.3 GPU-hours for both.

## vllm30056 (N1): PASS-with-fixes

- **Setup:** about 12 min (`setup_lock.sh` 407 s, model 18 s, stage 278 s). Freeze equal to the authors' (156 packages); all 1,673 vLLM files hash-identical.
- **Jobs:**

  | Job | GPU | Speedup | Verdict |
  |---|---|---|---|
  | calibration (24.3 min) | 0 | 1.7899 | **`fail:exact`** (below) |
  | scoring | 0 | 1.7909 | `ties_ref` |
  | scoring | 0 | 1.7852 | `ties_ref` |
  | scoring on empty caches | 1 | 1.7962 | `ties_ref` |
  | second calibration | 0 | 1.7895 | `ties_ref` |

  Fix rep 7.632-7.663 s, parent 13.57-13.79 s. Peak 20,110 MiB.
- **Correctness:** 24/24, log-probability difference 0, in every job on both GPUs, and also against the stored reference pass.
- **Why the first calibration failed.** One parent run's second rep differs from every other run: 8 of 64 requests diverge, at output positions 4, 4, 8, 8, 11, 17, 21 and 82, and all continue in free text. This is the engine's known non-repeatability on this load ([../findings.md](../findings.md), section 5). The scorer applies its one-rerun rule only in scoring and quick modes, not in calibration. The authors' own re-verification calibration also had a parent flip, but in an A/A run, which the gate does not check.
- **How often.** On the official load, on both GPUs (whose digests are identical to each other): parent 1 flipped run of 42 (1 of 126 reps); fix 0 of 19 runs; warm-ups 0 of 13. At about 1 run in 50, a 25-run calibration has about a **40% chance** of containing a flip.
- **Transfer loads:** 8 requests per rep 1.356 (card 1.347); no structured output 0.999 (card 1.000).
- **Variants:** `noadvance` 1.794 `fail:correct` (only the correctness pass catches it); `noparser` 1.779 `fail:exact,correct,settings`; `short` 2.326 `fail:counts,exact`. All as the card says.

**Fixes the tester asked for:**
1. In calibration mode, judge only fix against parent for the exact check and report parent-against-parent flips separately, or apply the one-rerun rule there too. Document it in `RUN.md`.
2. Say on the card that a fresh calibration contains a flip about 4 times in 10.
3. `RUN.md`'s time estimate (about 40 + 18 min) is stale; the card's (about 24 min per calibration, 9 min per scoring job) is right.

## sgl20503 (N2): PASS

- **Setup:** about 13 min. All 2,162 source files match; freeze equal to the authors' (171 packages); SGLang files identical apart from the documented generated `_version.py`. The manifest now also covers `PROVENANCE.md`.
- **Jobs:** calibration 1.0972 (27.1 min); scoring 1.0975 and 1.0949; a scoring job on the other GPU 1.0864 (it ran while other packages were being staged at low priority). All `ties_ref`, reward 1.0. Fix rep 9.38-9.71 s, parent 10.18-10.47 s. Single pairs ranged 1.048-1.106. Peak 20,544 MiB.
- **A/A:** 2.45% (0.991-1.0245).
- **Outputs:** digests identical everywhere, 0 flips; correctness 24/24.
- **Transfer loads:** `n4k` 1.011 (card 1.008); `r64` 1.067 (card 1.048).
- **Variants:** `prefix` 1.070, `ties_ref` (accepted by design, reward 0.87); `multistep` 1.091 `fail:settings`; `short` 1.202 `fail:counts,exact,correct`. All as the card says.
- **Optional:** the card could say that REF may read about 1.5% high on another host.
