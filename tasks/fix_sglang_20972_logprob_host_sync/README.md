# SGLang #20972: blocking logprob copy in the forward thread

With top-k logprobs requested, the parent makes a blocking GPU-to-CPU copy on every decode step. The human fix is
**1.142x** faster on an RTX 4090 (1.226x on an RTX 5090), with outputs bit-identical to the parent's. The 4090 band is
narrow (±0.60 %), and so is the noise (A/A at most 0.65 %). An independent tester rebuilt the package from an empty
directory and got the same result: PASS, no deviations.

Terms used below:
- **parent**: the unmodified tree, commit `e2b8463c`. **fix**: the human fix, commit `a27651d5`.
- **pair**: one parent run and one candidate run on the same GPU. A scoring job runs 3 pairs in ABBA order and
  reports the median pair. Pair speedup = parent time / candidate time.
- **A/A**: a parent/parent pair. Its distance from 1.0 is the noise floor.
- **REF**: the fix's speedup, the median over several jobs. **Band**: the range around REF that counts as a tie
  (`ties_ref`).
- **Gate**: the output checks. A candidate that fails any of them scores zero.

Hosts: "the 2-GPU server" is an RTX 4090 server with 2 GPUs, where this task was built and calibrated. "The 4-GPU
server" is a second RTX 4090 server, used for a second-server check of the screen. Both run driver 550.127.08 (CUDA
12.4) on a Threadripper PRO 5975WX.

## What is slow, and where does this task come from?

With `return_logprob` and `top_logprobs_num` > 0, the parent's sampler computes the top-k logprobs and immediately
calls `.tolist()` on them, and does the same for token-id logprobs. That is a blocking copy inside the forward
thread on every decode step, so SGLang's overlap scheduler cannot prepare step n+1 while the GPU runs step n. On this
load that costs ~0.7 ms per decode step of ~5.7 ms. The fix keeps the values on the GPU and converts them after the
step's copy-done event, for prefill and decode.

The PR was merged on 2026-03-27. We picked it from our list of upstream fixes mined from SGLang and vLLM history
(batch 4, prior probability of a stable gain 0.50), screened it, and then built the full task.

## What is the load and the score?

- Model: Qwen/Qwen2.5-1.5B-Instruct (bf16). Offline `sglang.Engine`, triton attention, pytorch sampling, overlap
  scheduler on (default), decode CUDA graphs up to batch 16, `mem_fraction_static` 0.80, seed 0.
- One `generate()` call of 8 requests x 128 random token ids -> 512 tokens, greedy, `ignore_eos`, every request with
  `return_logprob=True, top_logprobs_num=5`.
- **Metric: median wall time of 5 calls**, each with new seeded prompts (seeds 1-5, same in both arms; warm-up
  seed 1000).
- Score: parent / candidate per pair, median of 3 ABBA pairs. The harness also reports
  reward = min(1, (S - 1) / (REF - 1)).

## How much does the known fix gain?

Per-job medians: parent 2.888-2.936 s, fix 2.532-2.564 s per call.

| GPU | REF | Band | A/A max deviation | Jobs |
|---|---|---|---|---|
| RTX 4090 (2-GPU server, 2 cards) | **1.142** (median; range 1.137-1.148) | ±0.60 % (1.135-1.149) = max(0.5 %, 2 SD), SD 0.30 % | 0.58 % (calibration), 0.65 % (package test) | 7 |
| RTX 5090 | **1.226** (median of 3 jobs: 1.224, 1.235, 1.226) | not set; needs at least 1 % | 0.72 %; an A/A scoring job read 0.998 | 3 |

- The 4090 jobs: 1.142, 1.1475, 1.1368, 1.1435 on GPU1; 1.1388, 1.1412, 1.1431 on GPU0. GPU0 runs both arms ~0.7 %
  slower, so the ratio shows no card offset.
- A second calibration on GPU0, run after the band was set, gave 1.142.
- Second-server check of the screen (environment copied to the 4-GPU server, same load and seeds): 1.141 and
  1.139, median 1.140x, -0.2 %; output ids and top-5 logprobs identical to the 2-GPU server's.
- RTX 5090: the package runs unchanged. The gain is larger (1.22-1.24) because the removed copy is a fixed host cost
  and each decode step is shorter there. The three 5090 jobs spread 0.95 %, so the 4090 band is too tight; a 5090 gate
  should use REF ~1.226 and a band of at least 1 %. `package/harness/gate.json` still holds the 4090 REF, so on the
  5090 the fix reads `above_ref`.
- Transfer loads (one pair each): batch 1 1.173x; batch 32 1.162x; `top_logprobs_num` 20 1.157x; `return_logprob`
  without top-k **1.002x**. The last is a negative control: with no top-k there is nothing to remove, so a scoring
  load must ask for top-k logprobs.

## How are outputs checked?

All checks are in `package/harness/check.py`; thresholds in `package/harness/gate.json`.

1. **Exact outputs**: every timed rep's digest (output ids, text, sampled-token logprobs and top-5 logprobs rounded
   to 1e-3) equals the paired parent's. The fix is bit-identical in every rep of every job.
2. **Token counts** per rep.
3. **Settings readback**: the live `ServerArgs` equal the paired parent's, ports aside.
4. **Correctness pass**: 24 fixed word prompts, tokenized in the bench and sent as ids, in 3 `generate()` calls
   (batch 16, 1, 7), 64 greedy tokens, top-5 logprobs and token-id logprobs for 4 fixed ids (the second code path the
   PR changes). Ids identical; every logprob within 1e-3 nats; a top-k id may differ only at a tie within 1e-3.

**Why the correctness prompts are sent as token ids.** The first calibration sent them as text, and the fix then
failed its own correctness pass, deterministically: in the batch-7 call all 7 outputs had logprobs off by up to
0.25 nats and 3 of 7 diverged at exact ties, while the parent was bit-identical in all its runs. With text prompts
the 7 requests are tokenized one by one and reach the scheduler staggered. Most likely the fix's scheduler loop, no
longer blocked by the copy, groups them into prefill batches differently, and a different batch shape changes bf16
numerics (not instrumented). Sent as ids, all requests arrive together and the fix is bit-identical.

## Which broken or cheating versions are caught?

Python-only edits on the fix (`package/variants/apply.py`), one full scoring job each.

| Variant | What it does | Speedup | Verdict |
|---|---|---|---|
| `droplp` | decode-step top-k logprobs neither copied nor returned | 1.144x | `fail:exact,correct` (0/15 rep digests equal) |
| `stalelp` | each decode step returns the previous step's top-k (a naive "copy asynchronously, read one step later") | 1.141x | `fail:exact,correct` (top-5 values off by up to 16 nats) |
| `fp16lp` | top-k taken from an fp16 copy of the logprobs | 1.139x | `fail:exact,correct` (28 % of top-5 values off by more than 1e-3, max 3.8e-3 nats) |

None of the three is faster than the fix: on this load the blocking copy was the only large cost.

## What are the limits?

- **Exactness depends on how requests arrive.** With text prompts the human fix itself fails a bit-exact logprob
  rule (see above). The shipped loads send token ids. A candidate that changes request batching, or a load with
  staggered arrivals, would need a tolerance rule like #16521's.
- The timed check compares digests of logprobs rounded to 1e-3. A legitimate rewrite that changes a logprob by 1e-6
  can still flip one rounding boundary and fail `exact`. Not seen with the fix, but a near-equivalent candidate could
  be rejected; the correctness pass's 1e-3 tolerance does not have this edge.
- No audit hook, timing witness, integrity manifest or vault: the bench runs inside the candidate's process, and the
  settings readback reads the candidate's own `ServerArgs`.
- One host for calibration, two cards; no card offset in the ratio.
- Synthetic load (random prompt ids, 8 requests, top-5). Without top-k logprobs there is no gain.
- None of the broken variants was faster than the fix, so the checks were tested on wrong outputs, not on
  wrong-and-faster ones.

## What does it cost to run?

| Step | Time |
|---|---|
| Setup from an empty directory (build, model, CUDA pieces, stage) | ~23 min in the independent test |
| One scoring job (2 warm-ups, 2 correctness passes, 3 pairs) | 9.4-10 min of GPU time; ~9 min on the RTX 5090 |
| Calibration (25 runs) | ~25 min |
| Peak VRAM | 20.6 GB; 27.0 GB on the RTX 5090 |
| Disk | ~35 GB under the package |

One timed run is 59-61 s: 33 s engine start-up, an 11 s warm-up call, 5 timed calls of 2.5-2.9 s. Setup needs `g++`:
FlashInfer compiles its RMSNorm module at engine start with the CUDA 12.8 `nvcc` that the package downloads.

## Has someone else rebuilt it?

Yes. An independent tester copied the package alone into an empty directory and followed `package/RUN.md` without
edits. Verdict **PASS, no deviations**. This ran on the 2-GPU server, the authors' own host.

- Build: 2220/2220 source files, freeze (177 packages) and 2171/2171 sglang files equal the authors', both arms.
- Calibration REF 1.139, A/A max 0.52 %, exact digests 25/25, correctness 24/24.
- Scoring jobs 1.1453 (GPU0) and 1.1452 (GPU1), both `ties_ref`, every rep digest identical.
- Variants: `droplp` 1.146x, `stalelp` 1.145x, `fp16lp` 1.138x, all `fail:exact,correct` as the card says.
- Transfers: b1 1.163, b32 1.159, top-20 1.146, no top-k 0.997 (card 1.173 / 1.162 / 1.157 / 1.002).

The authors' own empty-directory test passed (scoring 1.1431, calibration 1.1412, A/A 0.65 %). The switch of the
correctness pass from text to token ids came from the authors' own calibration, not from that test, and was copied
into the test directory before its GPU steps.

## How to run it

```
cp -r package /data/sgl20972 && cd /data/sgl20972 && cp site.env.example site.env   # set STAGE, GPU, CPUS, PORT
bash setup/setup_lock.sh && bash setup/get_model.sh && bash setup/fetch_cuda.sh && bash setup/stage.sh
bash harness/run.sh cal_g0 cal && bash harness/run.sh s1_g0 score                  # calibration, then the fix
```

Every step, the expected output and the variant commands are in `package/RUN.md`.
