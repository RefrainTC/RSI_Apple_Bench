# Acceptance batch 6, part 2: vLLM #7822 and #16436, and the public repo's agent path (2026-11-25)

The batch-6 tester (see [batch6.md](batch6.md)) also gave the #7822 and #16436 packages their first independent test, on the 4-GPU 4090 server, GPUs 0 and 1. It then took the two tasks from the **public repo** and tried to score candidates by following only the agent-facing `instruction.md`.

## Summary

| Package | Verdict | REF: tester vs card | A/A max deviation: tester vs card | Main finding |
|---|---|---|---|---|
| iso7822 (vLLM #7822) | **PASS-with-fixes** | calibration 1.4807 vs 1.480; fix jobs 1.4799, 1.4714, 1.4758, 1.4809, 1.4770, median 1.4770 | 0.37% vs 0.16% (public-repo A/A job 0.17%) | one of 5 fix jobs fell 0.01% below the ±0.5% band; three variants the card names are not shipped |
| iso16436 (vLLM #16436) | **PASS** | calibration 1.0908 vs 1.093; jobs 1.0906, 1.0856, 1.0915, inside 1.0746-1.0964 | 0.47% vs 0.43% | none of substance |

GPU time: about 1.4 GPU-hours for #7822 and 2.5 for #16436.

## iso7822: PASS-with-fixes

- **Setup:** about 18.5 min (venvs 434 s, model 429 s for 8.5 GB, stage 245 s, workspace 7 s). Freeze and vLLM files identical to the authors'; manifest clean.
- **Calibration:** 15.4 min (card 14). REF **1.4807**. The output thresholds came out exactly as the card's (tie 0.8125 nats, minimum exact share 0.1902). Parent 9.116 s, fix 6.16 s. Peak 20,830 MiB. The venv-equivalence pair read 1.4376 (one pair, on a partly cold cache; not investigated).
- **Fix jobs:** 1.4799, **1.4714**, 1.4758, 1.4809, 1.4770. The 1.4714 job had pairs 1.481 / 1.3775 / 1.471: one candidate run was 7% slow (6.61 s against 6.16 s), which pulled the median 0.01% below the ±0.5% band (verdict `above_baseline_below_ref`, reward 0.981). The others are `ties_ref`. Between-job SD about 0.27% (card 0.17%). A scoring job takes 4.8-5.0 min.
- **The band is tight.** With ±0.5%, a single slow run can push a correct fix out of band; the tester estimates about 1 job in 5.
- **Transfer loads:** 10 output tokens 3.92× (card 3.93) and 300 repeats 1.55× (card 1.55) both end `output_mismatch`; the identity load 1.49× passes. The output thresholds are calibrated on the official load only, so on these two loads they reject the human fix.
- **Variants:** C1 2.737× `output_mismatch` and C5 1.461× `output_mismatch` plus a correctness failure, both as the card says. **C9, C11 and C12, which the card lists, are not shipped:** the variant script makes C1-C8 only.

**Fixes the tester asked for:** widen the band, or document that about 1 job in 5 lands just outside ±0.5%, or have the summary flag an outlier candidate run; ship or drop C9, C11 and C12; say on the card that the fix fails the output check on the two transfer loads.

## iso16436: PASS

- **Setup:** about 19 min (venvs 564 s, model 246 s, stage 313 s, workspace 13 s); manifest clean.
- **Calibration:** 73.8 min (card 73). REF **1.0908**, A/A 0.47%.
- **Scoring jobs** (about 19.3 min each): 1.0906 (GPU0), 1.0856 and 1.0915 (GPU1), all `ties_ref`. Median time per output token: parent 5.40-5.48 ms, fix 4.97-5.03 ms. Correctness 24/24; 97-99% of texts identical. Peak 21,094 MiB.
- **Variants:** C6 1.108× `output_mismatch` (empty text), correctness 0/24; C2 755× `output_mismatch` through the first-token latency check (card 761×). Both as the card says.

## Problems common to both packages

- The harness version these two packages use has no stop file.
- Job processes are found by a tag that does not include the package's root directory, so two copies run with the same labels on one host could kill each other's runs.
- The #16436 server port is fixed at 18430 + GPU and ignores `PORT` (`RUN.md` says so).
- The scoring script leaves compile-cache directories under the stage.

## Does the public repo match what was tested?

Yes, for scoring. Both public package manifests check clean. A file-by-file comparison against the tested packages shows only:
- placeholder paths in the freeze files (on the engine line only, which setup ignores) and in provenance files;
- a host name in the header comment of the CPU-pinning table;
- task READMEs that were dropped or translated into English (the integrity check excludes the task README);
- one behaviour change: a rule that protected the authors' home directory by user name lost that name. It has no practical effect off the authors' host, and the home-directory rule still applies.

## Can an agent score a candidate from `instruction.md` alone?

The scoring command works as written: `bash harness/score.sh tasks/<task> <label> <workspace dir>`.

| Candidate | #7822 | #16436 |
|---|---|---|
| Untouched parent workspace | 0.99995 `no_gain` (pairs 1.0002 / 0.9983 / 0.9999) | 1.0022 `no_gain` |
| Human fix copied into the workspace's engine tree | 1.4770 `ties_ref` (2 files, +22/-10 lines) | 1.0915 `ties_ref` |

Job times, warm-ups and pair counts matched the instruction. But the agent-facing text had five problems:
1. **It never says where a workspace comes from.** (`RUN.md` step 5 builds it into `tasks/<task>/workspace`.)
2. **The agent's quick-measure script silently measures the wrong tree.** `dev_run.sh` calls its helper with the default workspace `tasks/<task>/workspace`. Run from a copy of the workspace, it measures the pristine task workspace instead and writes its results there. A copy holding the correct fix reported a 9.12 s first pass, 67,870 computed prefill tokens and a speedup of 0.999: **an agent working in a copy sees "no gain" for a correct fix.** Official scoring is not affected. #16436's `dev_run.sh` is built the same way (not run). Fix: pass the script's own directory as the workspace, or tell the agent to work in place.
3. The workspace carries its own `instruction.md`, an older text with a different title and no Scoring, Measuring or Rules sections.
4. There are two git repositories: the engine tree (committed at the pristine state) and the workspace root, which ignores the engine tree. `git diff` from the root shows nothing.
5. The instruction says "1.0× scores nothing", but the reward is continuous (the #16436 A/A job got reward 0.024 with verdict `no_gain`).

## Fixed after acceptance (2026-11-25)

No scoring code changed in either package. Neither fix has been re-run on a GPU yet.

**iso7822.**
- **`dev_run.sh` (problem 2 above).** The workspace generator (`harness/build_workspace.sh`) now writes a `dev_run.sh` that passes its own directory as the workspace, so a copy of the workspace measures itself and keeps its own `dev_runs/`.
- **The workspace's `instruction.md` (problem 3).** When the package sits as `package/` inside the public task directory, the workspace now gets the task's public `instruction.md`. A package copied elsewhere still gets its own shorter task text. The public instruction no longer asks the agent to work in place.
- **Canaries C9, C11 and C12.** These are now shipped as `scripts/make_canaries_r4f.sh`, the authors' own script that made C5v, C9-C12, V1 and V2. Only its paths were made portable. Run against the authors' task directory, it rebuilt all seven canary workspaces byte for byte. Their scores have not been re-run from the package.
- **Validation (static).** We checked shell syntax and did a dry run of the generator in scratch copies (public layout and stand-alone, with a fake engine). The generated `dev_run.sh` passed the copy's own directory when started from another directory, from inside the copy and through a symlink. The pristine checksum list verified.
- **Manifest:** `815682eda8ab...` became `95edddeea150...`.

**iso16436.**
- The same `dev_run.sh` and `instruction.md` change, with the same validation.
- **Manifest:** `dcb1427b52c1...` became `761c791530c0...`.

Not changed: the ±0.5 % band of #7822, the output check on the transfer loads, and the four problems common to both packages listed above.
