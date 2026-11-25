"""offline_c5y.py <out dir> : workload and correctness pass of the vLLM #35442 task (trusted, ours). The timed part is the
batch-5 screen's bench/offline_b5.py default mode (PTYPE=ids, no FIRST), unchanged: same prompts, seeds, digests, metric.

Timed mode (default): LLM(model=M, **LLM_KW). Each timed rep has its own seeded prompt set (seed 1 + rep, identical in both
  arms): N prompts of IN random token ids -> OL greedy tokens (ignore_eos) in one generate() call. One untimed warm-up
  before (prompt set of seed 1000, at most 8 prompts, 8 tokens). Metric: median rep wall time. Writes latency.json
  (per-rep token counts and sha256 of the output ids, as the screen), settings.json (the engine's resolved configuration)
  and ids.json (output ids of every request of every rep).
CORRECT=1: untimed correctness pass. 24 fixed text prompts (20-181 words), one generate() call each, 64 greedy tokens
  (EOS allowed to stop), with the produced token's logprob -> correct.json.
PTYPE=text (transfer load): N natural-language essay requests per rep instead of random ids; SPECSTATS=1 records vLLM's
  speculative-decoding counters after the timed reps (acceptance), in latency.json "spec_decode".
Env: M N IN OL REPS MEMU LLM_KW CORRECT PTYPE SPECSTATS."""
import hashlib, json, os, random, statistics, sys, time

WORDS = ("the of and to in is was for on that with as by at from his her it an were are which this be or has had "
         "river city music garden winter science market history paper light").split()


TOPICS = ["the history of river trade in medieval Europe", "how winter affects city gardens", "the science of light and colour",
          "the early history of paper making", "how markets set prices", "the music of the Baroque period",
          "how bridges are designed", "the life cycle of stars", "the history of public libraries", "how the human heart works"]
TOPIC_TMPL = "Write a long, detailed essay about {}. Use several paragraphs and explain each point carefully."


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
    from vllm import LLM, SamplingParams
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "1")); IN = int(E("IN", "256")); OL = int(E("OL", "512")); REPS = int(E("REPS", "3"))
    kw = {"gpu_memory_utilization": float(E("MEMU", "0.80")), "seed": 0}
    if E("LLM_KW"): kw.update(json.loads(E("LLM_KW")))

    def promptset(seed):
        r = random.Random(seed)
        if E("PTYPE", "ids") == "text":   # transfer load "text": natural-language requests (MTP drafts get accepted)
            return [TOPIC_TMPL.format(TOPICS[(seed * 7 + i) % len(TOPICS)]) for i in range(N)]
        return [{"prompt_token_ids": [r.randrange(100, 30000) for _ in range(IN)]} for _ in range(N)]

    t0 = time.perf_counter(); llm = LLM(model=M, **kw); startup = time.perf_counter() - t0
    json.dump(settings(llm), open(os.path.join(out, "settings.json"), "w"), sort_keys=True, default=str)
    if E("CORRECT", "0") == "1":
        r = random.Random(7)
        texts = [" ".join(r.choice(WORDS) for _ in range(20 + 7 * i)) for i in range(24)]
        sp = SamplingParams(temperature=0, max_tokens=64, logprobs=0)
        outs = []
        for t in texts:   # one prompt per call
            o = llm.generate([t], sp, use_tqdm=False)[0].outputs[0]
            lp = []
            for tok, d in zip(o.token_ids, o.logprobs or []):
                lp.append(d[tok].logprob if d and tok in d else None)
            outs.append({"ids": list(o.token_ids), "lp": lp, "text": o.text, "finish": o.finish_reason})
        json.dump(outs, open(os.path.join(out, "correct.json"), "w")); print("CORRECT_DONE", len(outs), flush=True)
        return
    sp = lambda ol: SamplingParams(temperature=0, max_tokens=ol, ignore_eos=True)
    t = time.perf_counter(); llm.generate(promptset(1000)[:8], sp(8), use_tqdm=False); warm = time.perf_counter() - t
    ts, ntoks, shas, saved = [], [], [], []
    for rep in range(REPS):
        p = promptset(1 + rep)
        t = time.perf_counter(); res = llm.generate(p, sp(OL), use_tqdm=False); dt = time.perf_counter() - t
        ids = [list(x.outputs[0].token_ids) for x in res]
        ts.append(dt); ntoks.append(sum(len(x) for x in ids)); shas.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16])
        saved.append(ids)
    spec = {}
    if E("SPECSTATS", "0") == "1":   # transfer load "text": vLLM's speculative-decoding counters (needs disable_log_stats false)
        try:
            for m in llm.get_metrics():
                if "spec_decode" in m.name: spec[m.name] = getattr(m, "value", None) if not hasattr(m, "values") else list(m.values)
        except Exception as e:
            spec = {"error": repr(e)}
    d = {"spec_decode": spec, "startup_s": startup, "warm_s": warm, "per_rep_s": ts, "median_s": statistics.median(ts), "metric_s": statistics.median(ts),
         "out_tokens_per_rep": ntoks, "out_sha_per_rep": shas, "M": M, "N": N, "IN": IN, "OL": OL, "REPS": REPS,
         "LLM_KW": json.loads(json.dumps(kw, default=str))}
    json.dump(saved, open(os.path.join(out, "ids.json"), "w"))
    json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps({k: d[k] for k in ("per_rep_s", "median_s")}), flush=True)


if __name__ == "__main__":   # vLLM may start the engine core with spawn, which re-imports this module
    main()
