# Reference solution: two stock vLLM flags

```
--max-num-batched-tokens 12288
--mamba-ssm-cache-dtype float16
```

That is the whole reference (`solution/ref_fp16/submission/extra_args.txt`; `env.txt` is empty; no change to
`repo/`). There is no upstream PR: this is our own task, and the reference was found by config sweeps over the token
budget, max-num-seqs, text-only mode, API servers, stream interval, attention backend and the Gated DeltaNet (GDN)
state dtype. It was the best of four finalists in one session (1.315). It is **not the ceiling**: fp16 state +
`--max-num-batched-tokens 16384` + Triton attention scored 1.321 in the same session (1 session, 23,494 MiB).

Why it helps (scheduler counters, baseline → reference, one 4090 round):
- 1024/64: the larger token budget cuts the steps that carry a prefill chunk from 271 to 57 (all steps 373 → 282).
- 128/128: fp16 GDN state halves the 51.9 MB per-request state, so all 256 requests run together (steps 924 → 533,
  median TTFT 5.6 → 1.4 s at the same 43 ms TPOT). The larger budget alone, with fp32 state, leaves TTFT at 5.8 s.
- A decode-step timing fit puts the per-request fp32 vs fp16 cost gap at about 3/4 of the expected state traffic.

fp16 GDN state is allowed by the task. No gated measurement flags it: public-probe drift is 0.0057 for both fp16 and
pristine (equal by construction, since the probes never read stored state); a 1024-token decode gives |Δlp| within
about 18% of pristine on the first 64 tokens; witness |z| < 1.

How to build the reference arm (after setup, see README.md section 9):
```bash
bash $H/mkstage.sh qwen35_4b_1x4090_hc ref_fp16 "--max-num-batched-tokens 12288" "--mamba-ssm-cache-dtype float16"
```
