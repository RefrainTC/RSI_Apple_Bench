# Acceptance batch 6: vllm30056 (N1) and sgl20503 (N2): INTERIM

**Status: the independent test is still running.** This page records what is known so far. The verdicts below are provisional and will be replaced by the final report.

The tester follows the same method as in batches 4 and 5: each package copied alone into an empty directory, `RUN.md` followed literally with fresh downloads, on RTX 4090s.

## Provisional results

| Package | Calibration speedup (card REF) | Calibration A/A max deviation | Provisional verdict |
|---|---|---|---|
| sgl20503 (SGLang #20503, waiting-queue set) | **1.097** (1.08 ± 4%), `ties_ref` | 2.45% (gate limit 3%) | **PASS** (provisional) |
| vllm30056 (vLLM #30056, end-of-reasoning scan) | **1.790** (1.79 ± 3%) | 2.22% | **PASS-with-fixes** (provisional) |

## sgl20503

The calibration lands inside the band, with an A/A spread under the 3% limit. This task is weak by design: an 8% effect against an A/A noise of 2-3%, which the card states.

## vllm30056: why "with fixes"

- The speed reproduces: the calibration's 1.790 matches REF 1.79.
- But **the fresh-install calibration ended `fail:exact`.** The cause is a single flip in a parent run: one rep whose outputs differed from the other parent runs.
- This engine is known not to be perfectly repeatable on this load. In the authors' runs, one run of each arm (1 of 69 parent runs, 1 of 38 fix runs) had one flipped rep, and a control load that runs neither the fix nor any grammar flipped too. See [../findings.md](../findings.md), section 5.
- Scoring mode handles this with a one-rerun rule: when counts, settings and correctness all pass and exactly one rep of one run differs, the job is repeated once and the rerun's verdict is final. **Calibration mode has no such rule**, so a single engine flip fails the calibration.
- The fix needed: apply the same rerun rule (or a stated tolerance for one flipped rep) in calibration mode, and say so in `RUN.md`.

The final report will add the scoring jobs, the variants and any further findings.
