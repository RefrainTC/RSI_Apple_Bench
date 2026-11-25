# Acceptance batch 1: task C, Apple MM and Apple ST on the RTX 4090, Apple MM on the RTX 5090 (2026-11-24)

An independent tester followed each sample card literally, in a fresh directory, without the authors' working directories. These were the first cards, written before the self-contained packages existed.

## Summary

| Card | Verdict | Reference (oracle) | A/A |
|---|---|---|---|
| Task C on the 4090 (run on the second, 4-GPU server) | **PASS-with-fixes** | 1.3236 and 1.3295, about 1% above the card's 1.312-1.314 from the other server | 0.9981 |
| Apple MM on the 4090 | **PASS-with-fixes** | 1.3906 (card 1.373-1.401) | 0.9953 (card 0.990-1.021) |
| Apple ST on the 4090 | **PASS-with-fixes** | 1.1411 (card 1.142-1.146) | 0.9748 once (a client token-count drop), then 1.0027 |
| Apple MM on the 5090 | **PASS-with-fixes** | 1.542 (card 1.549-1.560) | 1.003 |

## Task C (`qwen35_4b_1x4090_hc`) on the 4-GPU 4090 server: PASS-with-fixes

- **Reference:** 1.3236 (rounds 1.3217 / 1.3255) and, in a second run, 1.3295 (rounds 1.3258 / 1.3331).
  - Per workload: 1.345 / 1.456 / 1.184 and 1.351 / 1.462 / 1.190; the card says about 1.33 / 1.45 / 1.18.
  - The card claims 1.3120-1.3144 on the 2-GPU server. **This server reads about 1.0-1.3% higher.** That is outside the card's 0.16% A/A band, but still far above noise. The reference is host-specific and should be re-measured on each host.
  - Verdict `scored`. Correctness 0.998 / 1.0. Drift 0.0056 / 0.0063. Witness |z| ≤ 1.3. Peak 23,424 MiB.
- **A/A:** 0.9981 (rounds 0.9995 / 0.9967); worst workload 0.9952. Peak 22,628 MiB.
- **Time per scoring:** 1084-1119 s, matching the card.
- **Setup:** about 50 min, including 2 failed compile-cache builds and 2 scorings voided by audit false positives. A clean path would take about 10 min.

**Fixes needed:**
1. No from-scratch path: the card copied the authors' prebuilt venv and weights from shared storage. The build scripts (`setup_venv.sh`, `download_models.sh`, `build_pristine.sh`, `build_jit_master.py`) were never mentioned.
2. The scoring harness version and the correctness reference (`corr_ref.json`) were not in the repo.
3. Paths were hard-coded to the authors' directories in the library module and every shell script.
4. GPU and CPU slots fit only the 2-GPU server (the "other GPUs" check looked at GPUs 0 and 1 only).
5. **The venv was unpinned.** A fresh install pulled transformers 5.24 / tokenizers 0.23 / huggingface-hub 2.4, and vLLM 0.19.1 then failed at startup (`<|image_pad|> … no images were passed`). Fix: install the authors' recorded freeze.
6. The warm compile cache was undocumented; without it each arm starts cold. Its build script reported success even when every bench run failed.
7. **Audit false positive.** With a uv 0.11 venv, `venv/bin/python` goes through a symlink, so the audit (which used `abspath`, not `realpath`) logged vLLM's read of a CUDA library directory as 150 `read_protected` events. The verdict became `harness_fault` and the score void. Fix: use `os.path.realpath`.
8. The model revision was not pinned.
9. Interrupting the server during startup left vLLM processes behind.

## Apple MM on the 4090: PASS-with-fixes

- **Oracle:** 1.3906, reward 0.391; the card says 1.373-1.401.
  - Per shape: 1.599 / 1.393 / 1.207.
  - Guards: replay 1.012, text 0.999. Outputs 16,965 of 16,965 identical.
  - Compiled kernels: 218 pristine, 39 oracle, matching the card.
  - Peak 20,850 MiB. Wall time 1606 s, against the card's 1184-1260 s; a fresh box needed 4 warm-ups.
- **A/A:** 0.9953 (card 0.990-1.021). Per shape 0.997 / 0.999 / 0.990; guards 0.992 / 0.998.

**Fixes needed:**
1. The card's reproduce path copied the authors' venv, model and CUDA tree from shared storage. The real build recipe was two hops away, in a workflow note not written in English. Built fresh from it, setup took about 11 min, worked first time, and the pristine tree's digest matched.
2. The memory guard needs a venv path that is not a symlink (it matches processes through `/proc/*/maps`).
3. The seed and commands reached the card only late; before that the tester used the workflow note.
4. The card should give the fresh-box pass time, about 27 min.

## Apple ST on the 4090: PASS-with-fixes

- **Setup** worked as written in about 5 min. The tester replaced two copy steps from the authors' directories with a fresh uv CPython and CUDA download. The lock check passed (215/215), and the harness diffs matched the card line for line.
- **Oracle: PASS.** 1.1411, reward 0.141; the card says 1.142-1.146.
  - Per point: 1.149 / 1.153 / 1.122.
  - Guards: c16 1.119, ttft_c1 1.045. Floor probe intact; outputs equivalent.
  - Peak 21,268 MiB, identical to the card. Wall time 1190 s.
- **First A/A: off.** 0.9748; the card says 0.998-1.003.
  - The whole gap was one point, c1 with 2000 output tokens, at 0.928.
  - One pristine request reported 1140 output tokens at an inter-token latency of 8.37 ms. Its neighbours reported 2000 tokens at 4.77 ms, and all three had the same 9.56 s latency. So the server was not slower: the client's token count dropped, because the client re-tokenizes streamed text.
  - The authors traced it (rate about 3 per 10,000 requests) and added a verifier edit that uses the server's token count. See [../selection/apple-37-tasks.md](../selection/apple-37-tasks.md).
- **Second A/A:** 1.0027, inside the card.

**Fixes needed:** the card set `TASK=`, but the scripts read `A4_TASK_DIR`, whose default was an author path; the copy steps from author directories had to become download commands; the card had to state the A/A tail risk.

## Apple MM on the 5090: PASS-with-fixes

- Fresh setup took 20.7 min, and each pass about 18 min.
- **Oracle 1.542**, reward 0.542. The card says 1.549-1.560, so this is 0.007 below the range, within the ±0.02 allowance.
  - Per shape: 1.845 / 1.568 / 1.267.
  - Replay guard median 0.960, text guard 1.001. Outputs 16,965 of 16,965 identical. Compiled kernels 218 / 39.
- **A/A 1.003.** Peak memory 28,472 MiB (27.8 GiB).

**Fixes needed:** scripts hard-coded the authors' paths; the 3 harness patches shipped only as already-patched files, with no diff; there was no model download step; the card and the workflow named different entry scripts; the A/A number has to be read from the round records, because its reward file is 0 by design.

## Tester's notes

- No elevated privileges were used.
- The tester briefly created a stray empty lock file in a shared lock directory and then deleted it.
- It used one GPU after checking that it was empty, before the authors confirmed it was free.
