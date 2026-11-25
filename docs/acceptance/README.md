# Acceptance tests: has someone else rebuilt each task?

**Short answer.** Every task with a package was rebuilt from an empty directory at least once, and most were also rebuilt by an independent tester who wrote none of the material. Every independent rebuild of a packaged task reproduced the authors' speedups within the task's band. The one exception is the Φ-Bench port, which is not packaged: its setup worked, but its reference score read 3-4% lower on the tester's quieter host. The tests also found real problems, most of them in the instructions (missing build steps, hard-coded paths) and two in the harness itself (a silent cache-copy failure, and an audit that voided honest runs). Those were fixed; the fixes are listed per report.

## Two kinds of test

- **Independent acceptance.** A tester (an agent that wrote none of the cards or packages) copies one package alone into an empty directory on a server, follows its instructions literally, and uses fresh downloads of everything: interpreter, wheels, source, models, CUDA. It does not read the authors' working directories, results or predictions until its own numbers are in. It reports a verdict:
  - **PASS**: followed as written, numbers inside the card's band, every broken variant caught;
  - **PASS-with-fixes**: numbers reproduce, but the instructions or harness needed fixes, which the report lists;
  - **FAIL on numbers**: the setup works but the scores do not match the card.
- **Authors' empty-directory runs.** The authors copy each package alone into an empty directory and follow its `RUN.md` with fresh downloads, before calling it ready. See [authors-verification.md](authors-verification.md).

Terms used in the reports: *REF* is the human fix's calibrated speedup; the *band* is the ± range around REF in which a candidate counts as tying the reference (`ties_ref`); *A/A* scores the unmodified code against itself; a *calibration* job measures REF and A/A from 5 AB and 5 A/A pairs; a *scoring* job scores one candidate with 3 ABBA pairs; *variants* are deliberately broken or cheating versions that the gate must reject.

## Which report covers which task?

| Task | GPU | Independent verdict | Report |
|---|---|---|---|
| Task C (`optimize_vllm_qwen35_4b_high_concurrency`) | 4090 (second server) | PASS-with-fixes | [batch1.md](batch1.md) |
| Task C | 5090 | PASS-with-fixes | [batch1-5090.md](batch1-5090.md) |
| Apple MM | 4090 | PASS-with-fixes | [batch1.md](batch1.md) |
| Apple MM | 5090 | PASS-with-fixes | [batch1.md](batch1.md) |
| Apple ST, adapted | 4090 | PASS-with-fixes | [batch1.md](batch1.md) |
| Φ-Bench vLLM scheduler | 5090 | FAIL on numbers (setup works after fixes) | [batch1-5090.md](batch1-5090.md) |
| SGLang #6994 | 4090 | PASS-with-fixes | [batch2.md](batch2.md) |
| vLLM #12380 | 4090 | PASS-with-fixes (an outsider was blocked until the scorer was shipped) | [batch2.md](batch2.md) |
| vLLM #14930 | 4090 | PASS-with-fixes | [batch2.md](batch2.md) |
| vLLM #13365 | 4090 | PASS-with-fixes | [batch2.md](batch2.md) |
| vLLM #14857 | 4090 | PASS-with-fixes | [batch2.md](batch2.md) |
| vLLM #7822, vLLM #16436, vLLM #14778 | 4090 | no independent rebuild of the package; authors' empty-directory run PASS | [authors-verification.md](authors-verification.md) |
| SGLang #16521, SGLang #20972 | 4090 | PASS | [batch4.md](batch4.md) |
| SGLang #9460, vLLM #37338, SGLang #27868, SGLang #14956 | 4090 (second server) | PASS | [batch4.md](batch4.md) |
| vLLM #35442, vLLM #27418 | 4090 | PASS-with-fixes; fixes applied and re-verified | [batch5.md](batch5.md) |
| vLLM #30056, SGLang #20503 | 4090 | in progress; provisional PASS-with-fixes and PASS | [batch6.md](batch6.md) |

The batch-2 report covers the ISO-Bench cards from both screening batches 2 and 3, so there is no separate batch-3 report. The cards tested in batches 1 and 2 were tested before the self-contained packages existed; their packages were later rebuilt by the authors from empty directories.

## What did the tests find, across batches?

1. **Early cards could not be followed from the card alone** (batches 1-2): commands pointed into the authors' directories, build steps and full commit hashes were missing, paths were hard-coded, and two scorers shipped only as a diff against a base the repo did not contain. This is why every task now ships as a self-contained package. From batch 4 on, every package was followed from `RUN.md` with no edits.
2. **Builds are reproducible.** Every rebuild that could be checked gave byte-identical engine files and identical dependency freezes, because the locks pin hashes and dependency cutoffs.
3. **Numbers reproduce, within bands.** Across batches 2-5, every human-fix scoring job landed in its card's band except one job 0.01% below on a busy host.
4. **Host noise is the main source of outliers**, usually from the tester's or authors' own parallel builds and copies on the same host.
5. **Two harness bugs were found only by independent testing**: an audit that voided honest runs through a symlinked venv (task C), and a cache copy that failed silently on a fresh install and broke the first calibration (vLLM #35442). See [../findings.md](../findings.md).
6. **REF depends on the host.** Task C reads about 1% higher on the second 4090 server, and Φ-Bench's ratios are lower on a quieter 5090. Measure REF on the machine you score on.
