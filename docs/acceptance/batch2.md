# Acceptance batch 2: five ISO-Bench cards (2026-11-24)

An independent tester followed five cards from ISO-Bench screening batches 2 and 3 from scratch, in fresh directories, without the authors' working directories, results or prediction files. Two cards were run by the lead tester and three by copies of it working under the same rules. All five ran on 2026-11-24 on both 4090 servers (the 2-GPU server's GPUs 0-1 and the 4-GPU server's GPUs 0-3). These cards were tested before the self-contained packages existed; the problems below are why every task now ships as one.

## Summary

| Card | Verdict | REF: tester (multi-job) vs card | A/A max deviation: tester vs card | Main blocker for an outside user |
|---|---|---|---|---|
| SGLang #6994 | PASS-with-fixes | 1.4831 (6 jobs; calibration 1.4749) vs 1.474 (calibration 1.475) | 0.62% vs 1.15% | scripts only in an untracked copy; no build steps |
| vLLM #12380 | PASS-with-fixes, outsider blocked | 1.516 (4 jobs; calibration 1.5145) vs 1.507 (calibration 1.513) | 4.2% vs 2.3% | base harness not in the repo; the exact scorer could only be rebuilt from the authors' copy |
| vLLM #14930 | PASS-with-fixes | 1.0899 (4 jobs; calibration 1.0892) vs 1.0909; one job 0.01% below the band | 0.28% vs 0.19% | harness shipped as a diff against a base that had moved on; about 40 min to reconstruct |
| vLLM #13365 | PASS-with-fixes | 1.0957 (4 jobs; calibration 1.0964) vs 1.0934 (calibration 1.0952) | 0.69% vs 0.47% | a missing data file made every fix job `gate_reject` |
| vLLM #14857 | PASS-with-fixes | 1.0645 (6 jobs; calibration 1.0643) vs 1.0648 (calibration 1.063) | 0.25% vs 0.33%; one 1.28 pair from a host stall | the venv build needs an undocumented `xgrammar==0.1.17` override |

Every card's numbers reproduce within its band (one #14930 job by 0.01% outside). The broken variants that were run behave as the cards say, and every build that could be checked is byte-identical to the authors' records. None of the five cards could be followed from the card itself.

## Problems common to all five cards

1. **No card could be followed from the card alone.** Every command pointed into the authors' server directory, and no card named an off-server copy of its scripts. The only copies were untracked mirrors in the authors' working repo, which a clone would not contain. The tester used them as the minimum needed to proceed.
2. **No from-scratch build section.** No card gave the full 40-character commit hashes together with the merge date that the venv builders need for the dependency cutoff. The tester took them from GitHub's public API or the ISO-Bench data.
3. **Hard-coded paths everywhere.** Every script had to be rewritten with `sed` before it could run in a fresh directory.
4. **Two scorers shipped as a diff against a base the repo did not contain** (#12380, #14930); the base had moved on, so patching it gave rejects and wrong files. #13365 shipped a full copy but missed one data file the gate reads.
5. **Venvs are reproducible.** Where it could be checked (all five), the rebuilt venvs' engine file hashes equal the authors' records, because the builders pin the dependency cutoff to the merge date plus 2 days.
6. **Host noise from concurrent setup work.** The testers' own parallel setups (venv builds, 7-20 GB copies, integrity hashing) produced the only out-of-pattern numbers: a 1.28 A/A pair (#14857), a 4.2% A/A (#12380), single bad pairs, and the #14930 job 0.01% under its band. Don't build or stage on a server while it runs timed jobs.

## SGLang #6994 (LoRA batch metadata): PASS-with-fixes

- **REF.** Calibration on the 2-GPU server's GPU0: **1.4749** (5 pairs, 1.4704-1.4866); the card says 1.475 (1.468-1.478). Five scoring jobs on 3 cards and 2 servers: 1.4803; 1.4860, 1.4898, 1.4800; 1.4885. All six jobs: median **1.4831**, range 1.4749-1.4898, between-job SD 0.39%. Every job is inside the card's band 1.474 ± 1.1% (1.458-1.490), but the median is 0.6% above the card's and two jobs sit at the top edge. The card's explanation holds: the spread comes from the fix side (fix median rep 2.441-2.456 s on two cards, 2.475-2.481 s on a third; parent 3.62-3.69 s everywhere). A card the authors never used behaves like one of theirs.
- **A/A** (calibration, 5 pairs): 0.9972-1.0062, max deviation **0.62%** (card 1.15%).
- **Transfer loads** (one pair each): batch 16 **1.402** (card 1.40), batch 128 **1.161** (card 1.17), two adapters **1.478** (card 1.475).
- **Outputs.** Correctness pass, fix against parent: 24/24 identical token ids in all 6 jobs. Timed runs: token counts identical in every rep; text identical in 69 of 100 fix-against-parent reps and 18 of 25 parent A/A reps (card: 65% and 80%). The card's warning holds: timed outputs are not reproducible even parent against parent, so an output check on them must be tolerance-based.
- **Cost.** Scoring job 274-293 s of GPU time (card 269-282 s); calibration 813 s (card 797 s); peak 20,780 MiB (card 20.9 GB).
- **Setup time:** about 11-12 min for the first server, not counting the time to find the scripts and rewrite their paths.
- **Variants:** none exist for this card (the card says so).

**Fixes needed:** the scripts lived only in the authors' directory (7 scripts and 5 environment files needed path rewrites); no build commands (full hashes, the merge date 2025-06-11, the model download, the LoRA-adapter generator's arguments); a staging step depended on an interpreter copy made by a script not mentioned anywhere. Build reproducibility is good: the installed SGLang file hashes of both venvs equal the authors' records, and exactly the two LoRA files differ, as the card says.

## vLLM #12380 (`decode_only` reset): PASS-with-fixes, but not reproducible by an outsider until the base harness is shipped

- **REF.** Calibration: **1.5145** (1.513-1.522); card 1.513. Four scoring jobs on both servers: 1.523, 1.497, 1.511, 1.521; REF **1.516**, between-job SD 0.80% (card 1.507, SD 0.54%). All four inside 1.507 ± 1.1% (1.490-1.524), two near the top edge.
- **A/A:** calibration max deviation **4.2%** (one parent run at 2.292 s against 2.18-2.22 s), above the card's 2.3%; the card itself says all the noise is in the CPU-bound parent. The pristine workspace scored as a candidate: **0.998** (card 0.998), `no_gain`.
- **Variant K1** (truncate at 100 tokens): raw 1.95× (card 1.92×), caught as `output_mismatch`.
- Outputs bit-identical in every job; peak VRAM 20.1 GB in every job.
- **Setup:** 8.0 min build plus 4.6 min staging per server, plus about 15 min to reconstruct the harness and adapt paths.
- **Blocking fix: the scorer was not shipped.** The repo had the task files, but the harness only as a diff against a base that was not in the repo; the newer harness in the repo gave 4 rejects and 13 of 39 wrong files. The exact scorer (39 of 39 files byte-identical to the authors' manifest) could be rebuilt only from a copy of the authors' own harness, which an outside user does not have. The numbers above are valid because the scorer matched the authors' manifest.
- **Other fixes:** no from-scratch section (full hashes, merge date 2025-01-24, setup scripts, workspace steps, model revision); 23 scripts hard-coded the authors' paths; the CPU pinning table had GPU0 only and there were no per-GPU ports; the first job on a newly staged server was voided as `audit_flag` because transformers writes `hf/hub/version.txt` (create it at staging); with the harness's default ±0.5% band one human-fix job got reward 0.966, so the card's ±1.1% band must be configured.

## vLLM #14930 (Triton rejection sampler): PASS-with-fixes (one job 0.01% below the band)

- **REF.** Calibration: **1.0892** (1.0846-1.0920); card 1.0909 (1.087-1.096). Four scoring jobs: 1.0897, 1.0913, 1.0902, **1.0854**; multi-job REF **1.0899**, SD 0.24% (card 1.0909, SD 0.20%). All four verdicts `ties_ref`. The 1.0854 job is 0.01% below the card's ±0.5% band (1.0854-1.0964); it ran while other testers' jobs and setups were on that server, and took 20.8 min instead of 13. With a band of max(0.5%, 2 SD) from only 6 of the authors' jobs, ±0.5% is tight for a host that is not quiet.
- **A/A:** calibration max deviation 0.28% (card 0.19%). The pristine workspace as a candidate: **0.9986**, `no_gain` (card 0.9985).
- **Variant C1** (skip verification): 1.118×, caught as `output_mismatch`, 40% identical in the correctness run, exactly as the card says.
- **Prompt-pool re-screen** (parent only, 704 candidate prompts): 674 identical in all 3 batch orders (card 674); the first 464 selected prompts match the card's set in 463 of 464. The curated pool is reproducible.
- **Cost / memory:** 13.1-13.3 min per scoring job (card about 13), calibration 42 min (card 40), peak 20,282 MiB = 82.6% (card 82.6%).
- **Fixes needed:** no from-scratch path, and the harness shipped only as a diff against a moved base (a first attempt built that way was not the card's scorer and was set aside; the tester rebuilt it from the #13365 harness copy plus the diff, about 40 min of reading, giving 47 of 50 files byte-identical to the authors' manifest, the other 3 being a change note, a generated manifest and the pinning table); the calibration script re-summarises without sourcing the task's environment, so every calibration pair carried a false `audit` flag (REF unaffected); hard-coded paths, no per-GPU `VLLM_PORT`, no merge date or full hashes.
- **Not tested:** transfer loads, variants C2/C3, the torch-only rewrite.

## vLLM #13365 (Numba n-gram proposer): PASS-with-fixes

- **REF.** Calibration: 1.0964 (card 1.0952). Four scoring jobs on 3 cards and 2 servers: median **1.0957**, range 1.0926-1.0979, SD 0.21% (card 1.0934, 1.0916-1.0961, SD 0.15%). All four inside the card's ±0.5% band (1.0879-1.0989), all `ties_ref`; the median reads 0.21% higher.
- **A/A:** calibration max deviation 0.69%; an extra 3-pair A/A under scoring conditions 0.9959 / 0.9980 / 1.0014.
- **Variant C4** (search cap outside `propose()`): raw 1.095, caught only by the correctness group with distant matches (76 against 346 proposed draft tokens), exactly as the card says.
- **Cost / memory:** 9.4-9.7 GPU-min per scoring job (card 9.6), peak 20,008 MiB. One server's parent is 1.2% slower than the other's while the ratio is unchanged, as the card says.
- **Setup:** about 24 min from a clean start to the first GPU run.
- **Fixes needed:** no setup path (paths rewritten in 48 files); 8-character hashes and no merge date; **the copy was missing one data file (`client_stdlib_py310.json`); without it the gate crashed and every human-fix scoring job ended as `gate_reject`**; the base-venv recipe (parent plus numba/llvmlite) was only in a setup script the card never names; the scoring script never takes its GPU lock and leaves each job's compile cache behind; on one server the task environment pinned every job to one GPU's CPUs; `vllm.__version__` reads `dev`, not 0.7.3.dev215.
- **Build reproducibility:** the 624 parent vLLM source files are byte-identical to the authors' record.

## vLLM #14857 (Mamba2 prefill syncs): PASS-with-fixes

- **REF.** Calibration on the 4-GPU server: **1.0643** (5 pairs, 1.0629-1.0661); card 1.063. A second calibration on the 2-GPU server: **1.0647**. Four scoring jobs on 3 cards and 2 servers: 1.0649, 1.0681, 1.0638, 1.0628. All six jobs: median **1.0645**, range 1.0628-1.0681, SD 0.17% (card 1.0648, 1.0614-1.0656, SD 0.14%). Every job is inside 1.0648 ± 0.5%. As the card says, absolute times depend on the card (fix 5.37 s on one, 5.40-5.44 s on others) but the ratio does not.
- **A/A.** Second-server calibration: 0.9989-1.0025, max deviation **0.25%** (card 0.33%). First-server calibration: 1.0009, 1.0425, **1.2823**, 0.9993, 0.9991. The 1.28 pair is one parent run with reps 5.94 / 9.94 / 12.50 / 7.31 / 6.24 s; in that run the 100 ms GPU sampler itself had a ~5 s gap, so the whole host stalled, while three sibling acceptance setups were building venvs, copying 7-20 GB and hashing on the same server. The rerun on a quieter host is clean, so this is host noise, not the card. But it shows that the card's "median of 5 reps" does not protect a run in which 3 of 5 reps are hit; only the job's median of 3 pairs does.
- **Transfer loads** (one pair each): 256-token prompts **1.022** (card 1.020), 64 output tokens **1.019** (card 1.019), `max_num_seqs` 16 **1.031** (card 1.033).
- **Outputs.** Correctness pass 24/24 identical in all 6 jobs. Timed reps: token counts and outputs identical to the paired parent in every rep, and in all 50 A/A reps (bit-deterministic).
- **Cost.** Scoring job 516-605 s (card 521-589 s); calibration 1454-1535 s (card 1456 s); peak **22,122 MiB** in every job.
- **Setup time:** about 19 min for the first server.
- **Fixes needed:** scripts only in the authors' directory; **the venv build fails as the card implies**, because xgrammar 0.1.16 was removed from PyPI, and needs the override `xgrammar==0.1.17`, which the card did not mention (the authors' own override file shows they used it); no full hashes or merge date (fix `296f927f2493908984707354e3cc5d7b2e41650b`, parent `0032903a5bb7c7c655f52f4efdfcc221947e9ca8`, merged 2025-03-21). Build reproducibility is good: both venvs' vLLM file hashes equal the authors' records, and only `mamba_mixer2.py` and `_version.py` differ.

## Tester's notes

- No elevated privileges; separate ports per card and GPU; per-job GPU locks; stop files checked before every job (none ever appeared). No processes or sessions remained afterwards.
- For #12380, rebuilding the exact scorer required authors' material; that is recorded above as the finding that blocks an outside user.
- The server clocks differ by about 2 minutes; all times are UTC from each server's own clock.
