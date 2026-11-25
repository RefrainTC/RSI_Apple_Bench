"""sgl_b3.py <out dir> : batch-3 offline screening benchmark for SGLang (trusted, ours). sglang.Engine with LoRA adapters;
N prompts of IN random token ids, greedy, max_new_tokens=OL, ignore_eos; every request uses adapter LORA_NAME (from
LORA_PATHS "name=dir,..."). One small untimed warm-up, then REPS timed generate calls on the same prompts. Writes
<out>/latency.json like offline_b3.py (sha over output texts). Env: M N IN OL REPS MEMF LORA_PATHS LORA_NAME SGL_KW.
LORA_NAME may list several adapters ("a,b"): request i uses adapter i mod k. Calibration options (off by default):
ROTATE=1: every timed rep gets its own seeded prompt set (seed 1 + rep; warm-up seed 1000). CORRECT=1: instead of the timed
part, 24 fixed text prompts in 3 generate() calls (batch 16, 1, 7), 48 greedy tokens, top-5 logprobs -> <out>/correct.json."""
import hashlib, json, os, random, statistics, sys, time
def main():
    import sglang as sgl
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "64")); IN = int(E("IN", "256")); OL = int(E("OL", "256")); REPS = int(E("REPS", "3"))
    loras = [x for x in E("LORA_PATHS", "").split(",") if x]; name = E("LORA_NAME", "")
    if not os.path.isdir(M):
        from huggingface_hub import snapshot_download
        M = snapshot_download(M, local_files_only=True)
    kw = dict(model_path=M, mem_fraction_static=float(E("MEMF", "0.80")), random_seed=0, log_level="error")
    if loras: kw.update(lora_paths=loras)
    if E("SGLANG_PORT_BASE"): kw["port"] = int(E("SGLANG_PORT_BASE"))   # distinct per host/GPU (b3_pairs.sh)
    if E("SGL_KW"): kw.update(json.loads(E("SGL_KW")))
    pset = lambda seed: (lambda r: [[r.randrange(100, 30000) for _ in range(IN)] for _ in range(N)])(random.Random(seed))
    ROTATE = E("ROTATE", "0") == "1"; CORRECT = E("CORRECT", "0") == "1"
    prompts = pset(1000 if ROTATE else 0)
    llm = sgl.Engine(**kw)
    sp = lambda ol: {"temperature": 0, "max_new_tokens": ol, "ignore_eos": True}
    names = [x for x in name.split(",") if x]
    lp = lambda n: [names[i % len(names)] for i in range(n)] if names else None
    if CORRECT:
        words = "the of and to in is was for on that with as by at from his her it an were are which this be or has had".split()
        r2 = random.Random(7)
        texts = [" ".join(r2.choice(words) for _ in range(20 + 7 * i)) for i in range(24)]
        outs = []
        for a, b in ((0, 16), (16, 17), (17, 24)):
            res = llm.generate(prompt=texts[a:b], sampling_params=sp(48), lora_path=lp(b - a), return_logprob=True, top_logprobs_num=5)
            for r in res:
                mi = r["meta_info"]
                outs.append({"ids": [x[1] for x in mi["output_token_logprobs"]], "text": r["text"],
                             "lp": [{str(t[1]): t[0] for t in d} for d in mi["output_top_logprobs"]]})
        json.dump(outs, open(os.path.join(out, "correct.json"), "w")); print("CORRECT_DONE", len(outs))
        llm.shutdown(); return
    llm.generate(input_ids=prompts[:8], sampling_params=sp(8), lora_path=lp(8))
    ts, ntoks, shas = [], [], []
    for rep in range(REPS):
        if ROTATE: prompts = pset(1 + rep)
        t = time.perf_counter(); res = llm.generate(input_ids=prompts, sampling_params=sp(OL), lora_path=lp(N)); ts.append(time.perf_counter() - t)
        ntoks.append(sum(r["meta_info"]["completion_tokens"] for r in res))
        shas.append(hashlib.sha256(json.dumps([r["text"] for r in res]).encode()).hexdigest()[:16])
    d = {"wall_s": sum(ts), "per_rep_s": ts, "median_s": statistics.median(ts), "out_tokens_per_rep": ntoks, "out_sha_per_rep": shas,
         "M": M, "N": N, "IN": IN, "OL": OL, "REPS": REPS, "ROTATE": ROTATE, "ENGINE_KW": {k: v for k, v in kw.items()}}
    json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps(d))
    llm.shutdown()
if __name__ == "__main__":
    main()
