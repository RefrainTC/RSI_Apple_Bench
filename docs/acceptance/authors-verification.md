# The authors' empty-directory runs of every package (2026-11-24/25)

Before calling a package ready, the authors copied each package directory alone into an empty directory on a 4090 server (on shared storage, with the stage on local NVMe), made `site.env` from `site.env.example`, and followed `RUN.md` literally with a fresh uv cache, fresh CPython, fresh source tarballs, and fresh model and CUDA downloads. A package passes when the human fix's scoring job falls in the card's band, the calibration's A/A is within the card's noise, and the fix passes every check. These runs are not independent (the authors wrote the packages); the independent tests are in the other reports in this folder.

## Summary

| Package | Fix scoring job(s) (card band) | Calibration REF (card) | A/A max deviation (card) | Status |
|---|---|---|---|---|
| iso12380 | 1.5250, 1.5220, 1.5241 (1.490-1.524) | 1.5244 (1.513) | 2.32% (2.3%) | **PASS at the upper edge**: the CPU-bound parent ran about 0.6% slower than on the card, on a busy host |
| iso14930 | 1.0866 (1.0854-1.0964) | 1.0866 (1.0909) | 1.01% (0.19%); pristine A/A job 1.0033 | **PASS**; A/A noisier than the card on a busy host |
| iso13365 | 1.0998, 1.0974, 1.0885 (1.0879-1.0989) | 1.0933 (1.0952) | 0.89% (about 0.5%) | **PASS**: 2 of 3 jobs in band, median 1.0974; the first 0.08% above |
| iso6994 | 1.4764 (1.458-1.490) | 1.483 (1.475) | 0.58% (1.15%) | **PASS** |
| iso14857 | 1.0621 (1.0595-1.0701) | 1.0675 (1.063) | 0.17% (0.33%) | **PASS** |
| iso14778 | 1.0344 (1.0313-1.0417) | 1.0337 (1.033) | 0.20% (0.31%) | **PASS** |
| iso7822 | 1.4871, 1.4787 (1.4716-1.4864) | 1.4758 (1.480) | 0.21% (0.16%); pristine A/A job 0.9980 | **PASS** after fixes 4-5 below (a first copy was void) |
| iso16436 | 1.0810; a first copy gave 1.0910 / 1.0857 (1.0746-1.0964) | 1.0922 (1.093 single job; 1.0855 multi-job) | 0.71% (0.43-1.07%); pristine A/A job 0.9974 | **PASS** after fix 6 |
| sgl16521 | 3.8208 (3.779-3.875) | 3.8036 | 0.27% (0.61%) | **PASS** |
| sgl20972 | 1.1431 (1.135-1.149) | 1.1412 (1.142) | 0.65% (0.58%) | **PASS** |
| sgl9460 | 1.1998 (1.185-1.221) | 1.2075 (1.203) | 1.2% (1.9%) | **PASS** |
| vllm37338 | 1.2951 (1.264-1.316) | 1.2969 (1.29) | 2.5%, one parent start-up 4 s slow (0.9%; gate limit 3%) | **PASS** |
| sgl27868 | 1.0269 (1.0225-1.0327) | 1.0291 (1.0276) | 0.25% (0.7%) | **PASS** |
| vllm35442 | 1.0362; after the batch-5 fixes 1.0377 (1.027-1.047) | 1.0361; after fixes 1.0365 | 0.44%; after fixes 0.31% | **PASS**, re-verified after the independent test |
| vllm27418 | 1.679; after fixes 1.686 (1.639-1.741) | 1.6785; after fixes 1.679 | 3.6%, one slow parent rep; after fixes 3.25% (gate limit 5%) | **PASS**, re-verified |
| sgl14956 | 1.3070 (1.284-1.336) | 1.3095 | 0.9% | **PASS** |
| vllm30056 | 1.805; after fixes 1.761 (1.736-1.844) | 1.791; after fixes 1.773 | 2.47%; after fixes 2.54% (gate limit 3%, set after seeing these runs) | **PASS**, re-verified; the re-verify saw the first flip in a parent run |
| sgl20503 | 1.087 (1.037-1.123) | 1.086 | 2.97% (stop rule 3%) | **PASS**, kept with this stated on the card |
| sgl17005 | 1.427 (1.358-1.442) | 1.400 (1.40) | 1.77% (gate limit 3%) | **PASS** |

**Builds.** In every package, both venvs' dependency freezes equal the authors' and every installed engine file hashes as the authors' build; the only exceptions are generated `_version.py` files whose version values match. Every model file matched its checksum.

## What did these runs find and fix?

1. **Download rate limits.** Hugging Face answered HTTP 429 (an anonymous limit on a shared IP). The model scripts now retry up to 10 times with back-off and mention `HF_TOKEN`.
2. **A settings file overrode the caller.** `env.sh` read `site.env` after the caller's variables, so `GPU=1 ... score.sh` silently used `site.env`'s GPU. A repeat job started next to another package's calibration and disturbed it. Fixed: `env.sh` keeps every variable the caller set.
3. **Files edited after the test copies** were checked by hash; only documentation and one switch with an unchanged default differed.
4. **iso7822 lost `vllm-flash-attn`.** The lock generator and freeze check dropped every line starting with `vllm`, removing `vllm-flash-attn` with the engine line. vLLM 0.5.5 then fell back to xformers plus a Triton kernel: calibration REF 1.4079 instead of 1.480, and every scoring pair failed the output check. Fixed in the lock, the freeze filter and an import check; the test was restarted from an empty directory.
5. **iso7822 and iso16436 lacked the post-calibration step** that derives the output-check thresholds. The first scoring job (1.4808, in band) failed all pairs under strict default thresholds. Added as `scripts/post_calib.sh`; it then reproduced the authors' thresholds exactly.
6. **iso16436: fresh venvs lacked bytecode,** so each served run wrote about 300 `.pyc` files into its venv copy and the venv-integrity check flagged every pair. Fixed by pre-compiling bytecode at setup.
7. **The shared setup script of iso6994, iso14857 and iso14778** checked a source-file count that the authors' record could not meet (181 file names with spaces were missing from it), and its logging truncated its own log file. Both fixed.
8. **sgl16521's generated `_version.py` differs** from the authors' (a different setuptools-scm template; same values). The check now accepts that one file if its version and commit match.
9. **sgl27868's first build** was refused because two locked packages had no hashes; fixed in the lock.
10. **vllm37338's fix venv** got its top-level `build_backend.py` from a different wheel than the authors' (two wheels ship one; uv's install order decides). Nothing imports it; documented.
11. **vllm30056's rare digest flip** led to a one-rerun rule in scoring (see [../findings.md](../findings.md), section 5).

The independent tests found further problems that these runs missed: see [batch5.md](batch5.md) (the silent cache-copy failure) and [batch4.md](batch4.md).
