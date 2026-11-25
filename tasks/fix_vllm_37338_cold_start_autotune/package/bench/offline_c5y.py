"""offline_c5y.py <out dir> : workload and teacher-forced output check of the vLLM #37338 task (trusted, ours; extends batch
4's bench/offline_b4.py FIRST=1 mode, whose prompts, seeds and digests are unchanged).

Timed mode (default): LLM(model=M, ...) then the FIRST generate() = N prompts of IN random token ids (seed 1) -> OL
  greedy tokens (ignore_eos), then REPS steady reps (seeds 2, 3, ...), same shape. Wall-clock stamps (time.time()) are
  written for: module start, import of vllm done, LLM() returned, first generate() returned, steady reps done. The
  harness turns them into the score: cold time to first result = first generate() returned - process launch (job.sh's
  start stamp), so imports, engine start-up and the first batch all count. Writes latency.json, settings.json (the
  engine's resolved configuration) and seqs.json (prompt seed and output ids of every request).
TF=1 (PARENT venv only): teacher-forced re-read. For every run dir in TF_DIRS (":"-separated) it rebuilds the prompts
  from their seeds, appends the produced output ids and runs one prefill with prompt_logprobs=1 (prefix caching off);
  for every output token gap = top-1 logprob - logprob of the produced token (0 = argmax) -> <out>/tf.json.
Env: M N IN OL REPS MEMU LLM_KW TF TF_DIRS."""
import time
T_MODULE = time.time()
import hashlib, json, os, random, statistics, sys


def prompts(seed, n, il):
    r = random.Random(seed)
    return [[r.randrange(100, 30000) for _ in range(il)] for _ in range(n)]


def settings(llm):
    """The resolved engine configuration (simple-valued fields of each sub-config of vllm_config)."""
    import dataclasses, enum
    out = {}
    try:
        vc = llm.llm_engine.vllm_config
        for name in ("model_config", "cache_config", "scheduler_config", "compilation_config", "parallel_config",
                     "load_config", "device_config", "speculative_config", "kv_transfer_config", "observability_config"):
            sub = getattr(vc, name, None)
            if sub is None: out[name] = None; continue
            fields = [f.name for f in dataclasses.fields(sub)] if dataclasses.is_dataclass(sub) else list(vars(sub))
            for f in fields:
                v = getattr(sub, f, None)
                if isinstance(v, enum.Enum): v = str(v)
                if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool, type(None))) for x in v): v = list(v)
                if isinstance(v, (str, int, float, bool, type(None), list)): out[f"{name}.{f}"] = v
                elif isinstance(v, dict): out[f"{name}.{f}"] = json.dumps(v, sort_keys=True, default=str)
    except Exception as e:
        out = {"error": repr(e)}
    return out


def main():
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "8")); IN = int(E("IN", "512")); OL = int(E("OL", "8")); REPS = int(E("REPS", "3"))
    TF = E("TF", "0") == "1"
    from vllm import LLM, SamplingParams
    t_import = time.time()
    kw = {"gpu_memory_utilization": float(E("MEMU", "0.85")), "seed": 0}
    if E("LLM_KW"): kw.update(json.loads(E("LLM_KW")))
    if TF:
        kw["enable_prefix_caching"] = False
        dirs = [d for d in E("TF_DIRS", "").split(":") if d]
        if any(s["in"] > 1024 for d in dirs for s in json.load(open(os.path.join(d, "seqs.json")))):
            # long prompts: the fp32 log-softmax over all prompt positions needs ~2 GB per 2k-token sequence, which does
            # not fit next to a 0.85 KV pool on 24 GB (transfer load "long"); smaller pool, one sequence per call
            kw["gpu_memory_utilization"] = min(kw["gpu_memory_utilization"], 0.6); os.environ["C5Y_TF_CHUNK"] = "1"
        tf_check(LLM(model=M, **kw), SamplingParams, out, [d for d in E("TF_DIRS", "").split(":") if d]); return
    t0 = time.perf_counter(); llm = LLM(model=M, **kw); startup = time.perf_counter() - t0; t_llm = time.time()
    json.dump(settings(llm), open(os.path.join(out, "settings.json"), "w"), sort_keys=True, default=str)
    sp = SamplingParams(temperature=0, max_tokens=OL, ignore_eos=True)
    seqs = []

    def timed(seed):
        p = prompts(seed, N, IN)
        t = time.perf_counter(); res = llm.generate([{"prompt_token_ids": x} for x in p], sp, use_tqdm=False); dt = time.perf_counter() - t
        ids = [list(r.outputs[0].token_ids) for r in res]
        seqs.append({"seed": seed, "n": N, "in": IN, "out": ids})
        return dt, sum(len(x) for x in ids), hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]

    first, ft, fs = timed(1); t_first = time.time()
    ts, ntoks, shas = [], [], []
    for rep in range(REPS):
        dt, nt, sh = timed(2 + rep); ts.append(dt); ntoks.append(nt); shas.append(sh)
    t_steady = time.time()
    d = {"t_module": T_MODULE, "t_import_done": t_import, "t_llm_done": t_llm, "t_first_done": t_first, "t_steady_done": t_steady,
         "startup_s": startup, "first_s": first, "first_tokens": ft, "first_sha": fs, "per_rep_s": ts,
         "median_s": statistics.median(ts) if ts else None, "out_tokens_per_rep": [ft] + ntoks, "out_sha_per_rep": [fs] + shas,
         "M": M, "N": N, "IN": IN, "OL": OL, "REPS": REPS, "LLM_KW": kw}
    json.dump(seqs, open(os.path.join(out, "seqs.json"), "w"))
    json.dump(d, open(os.path.join(out, "latency.json"), "w"))
    print(json.dumps({k: d[k] for k in ("startup_s", "first_s", "per_rep_s")}), flush=True)


HIST = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)   # the teacher-forced check counts output tokens whose gap exceeds each value (nats)


def tf_check(llm, SamplingParams, out, dirs):
    t0 = time.perf_counter(); res = {}
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    for d in dirs:
        seqs = json.load(open(os.path.join(d, "seqs.json")))
        items = []
        for rep, s in enumerate(seqs):
            for i, (p, o) in enumerate(zip(prompts(s["seed"], s["n"], s["in"]), s["out"])): items.append((rep, i, p, o))
        ch = int(os.environ.get("C5Y_TF_CHUNK", "0")) or len(items) or 1; rr = []
        for k in range(0, len(items), ch):
            rr += llm.generate([{"prompt_token_ids": p + o} for _, _, p, o in items[k:k + ch]], sp, use_tqdm=False)
        nonarg = []; maxg = 0.0; npos = 0; bad = 0; per_rep = {}; hist = {t: 0 for t in HIST}; sumg = 0.0
        for (rep, i, p, o), x in zip(items, rr):
            pl = x.prompt_logprobs
            if pl is None or len(pl) != len(p) + len(o): bad += 1; continue
            pr = per_rep.setdefault(rep, [0, 0, 0.0])
            for j, tok in enumerate(o):
                dct = pl[len(p) + j]
                if not dct or tok not in dct: bad += 1; continue
                top = max(dct.items(), key=lambda kv: kv[1].logprob)
                g = max(0.0, top[1].logprob - dct[tok].logprob); npos += 1; pr[0] += 1; sumg += g
                for t in HIST:
                    if g > t: hist[t] += 1
                if top[0] != tok: nonarg.append([rep, i, j, round(g, 4)]); pr[1] += 1
                maxg = max(maxg, g); pr[2] = max(pr[2], g)
        res[d] = {"all": {"n_pos": npos, "n_nonargmax": len(nonarg), "max_gap": round(maxg, 4), "len_mismatch": bad,
                          "n_gap_above": {str(t): n for t, n in hist.items()}, "mean_gap": sumg / max(npos, 1),
                          "per_rep": {str(k): [v[0], v[1], round(v[2], 4)] for k, v in per_rep.items()},
                          "nonargmax": sorted(nonarg, key=lambda z: -z[3])[:200]}}
        print("TF", d, (npos, len(nonarg), round(maxg, 4), bad), flush=True)
    json.dump({"dirs": res, "tf_s": time.perf_counter() - t0}, open(os.path.join(out, "tf.json"), "w"))


if __name__ == "__main__":   # vLLM >= 0.18 may start the engine core with spawn, which re-imports this module
    main()
