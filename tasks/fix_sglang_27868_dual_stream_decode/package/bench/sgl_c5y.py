"""sgl_c5y.py <out dir> : workload and correctness pass of the SGLang #27868 task (trusted, ours). The timed part is
batch 4's bench/sgl_b4.py default mode, unchanged: same prompts, seeds, digests and metric.

Timed mode (default): sglang.Engine, greedy, ignore_eos. N prompts of IN random token ids -> OL tokens in one
  generate() call. One untimed 8-request warm-up (seed 1000, 8 tokens), then REPS timed reps, rep k on prompt seed 1 + k
  (identical in both arms). Metric: median rep wall time. Writes latency.json (per-rep sha256 of output ids + text, as
  batch 4), settings.json (live ServerArgs) and ids.json (output ids of every request of every rep).
CORRECT=1: untimed correctness pass. 24 fixed text prompts (20-181 words), one generate() call each (batch 1, decode in
  CUDA graphs, where the dual-stream branch runs), 64 greedy tokens with the produced token's logprob and the top-5 logprobs -> correct.json.
Env: M N IN OL REPS MEMF SGL_KW CORRECT SGLANG_PORT_BASE."""
import dataclasses, hashlib, json, os, random, statistics, sys, time

WORDS = ("the of and to in is was for on that with as by at from his her it an were are which this be or has had "
         "river city music garden winter science market history paper light").split()


def main():
    import sglang as sgl
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "4")); IN = int(E("IN", "256")); OL = int(E("OL", "512")); REPS = int(E("REPS", "3"))
    CORRECT = E("CORRECT", "0") == "1"
    if not os.path.isdir(M):
        from huggingface_hub import snapshot_download
        M = snapshot_download(M, local_files_only=True)
    kw = dict(model_path=M, mem_fraction_static=float(E("MEMF", "0.80")), random_seed=0, log_level="error")
    if E("SGLANG_PORT_BASE"): kw["port"] = int(E("SGLANG_PORT_BASE"))
    if E("SGL_KW"): kw.update(json.loads(E("SGL_KW")))
    ids = lambda r, n: [r.randrange(100, 30000) for _ in range(n)]
    pset = lambda seed: (lambda r: [ids(r, IN) for _ in range(N)])(random.Random(seed))
    t0 = time.perf_counter(); llm = sgl.Engine(**kw); startup = time.perf_counter() - t0
    try:
        sa = dataclasses.asdict(llm.server_args)
        json.dump({k: v for k, v in sa.items() if isinstance(v, (str, int, float, bool, type(None)))},
                  open(os.path.join(out, "settings.json"), "w"), sort_keys=True)
    except Exception as e:  # recorded; the settings check then fails
        json.dump({"error": repr(e)}, open(os.path.join(out, "settings.json"), "w"))
    sp = lambda ol: {"temperature": 0, "max_new_tokens": ol, "ignore_eos": True}
    if CORRECT:
        r = random.Random(7)
        texts = [" ".join(r.choice(WORDS) for _ in range(20 + 7 * i)) for i in range(24)]
        outs = []
        # one prompt per call: a multi-prompt call reaches the scheduler in timing-dependent groups (the first prefill
        # can start before all prompts arrived), which changes batch shapes and therefore bf16 numerics run to run
        for a in range(len(texts)):
            for x in llm.generate(prompt=texts[a:a + 1], sampling_params=sp(64), return_logprob=True, top_logprobs_num=5):
                mi = x["meta_info"]
                outs.append({"ids": [t[1] for t in mi["output_token_logprobs"]], "lp": [t[0] for t in mi["output_token_logprobs"]],
                             "top": [[[t[0], t[1]] for t in d] for d in mi["output_top_logprobs"]], "text": x["text"]})
        json.dump(outs, open(os.path.join(out, "correct.json"), "w")); print("CORRECT_DONE", len(outs))
        llm.shutdown(); return

    def digest(res):
        return hashlib.sha256(json.dumps([{"ids": r.get("output_ids"), "text": r.get("text")} for r in res]).encode()).hexdigest()[:16]

    run = lambda seed, n, ol: llm.generate(input_ids=pset(seed)[:n], sampling_params=sp(ol))
    t = time.perf_counter(); run(1000, 8, 8); warm = time.perf_counter() - t
    ts, ntoks, shas, saved = [], [], [], []
    for rep in range(REPS):
        t = time.perf_counter(); res = run(1 + rep, N, OL); ts.append(time.perf_counter() - t); shas.append(digest(res))
        ntoks.append(sum(x["meta_info"]["completion_tokens"] for x in res))
        saved.append([list(x.get("output_ids") or []) for x in res])
    d = {"startup_s": startup, "warm_s": warm, "per_rep_s": ts, "median_s": statistics.median(ts), "metric_s": statistics.median(ts),
         "out_tokens_per_rep": ntoks, "out_sha_per_rep": shas, "M": M, "N": N, "IN": IN, "OL": OL, "REPS": REPS,
         "ENGINE_KW": {k: v for k, v in kw.items()}}
    json.dump(saved, open(os.path.join(out, "ids.json"), "w"))
    json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps({k: d[k] for k in ("per_rep_s", "median_s")}))
    llm.shutdown()


if __name__ == "__main__":
    main()
