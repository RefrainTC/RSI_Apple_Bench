"""offline_c5y.py <out dir> : workload and correctness pass of the vLLM #30056 task (trusted, ours). The timed part is the
batch-6 screen's bench/offline_b5.py default ids mode with SP_KW (as used for N1), unchanged: same prompts, seeds,
digests, metric.

Timed mode (default): LLM(model=M, **LLM_KW). Each timed rep has its own seeded prompt set (seed 1 + rep, identical in both
  arms): N prompts of IN random token ids -> OL greedy tokens (ignore_eos) in one generate() call, with SamplingParams
  extras from SP_KW ("logit_bias"; "json_schema" -> structured_outputs). One untimed warm-up before (prompt set of seed
  1000, at most 8 prompts, 8 tokens). Metric: median rep wall time. Writes latency.json (per-rep token counts and sha256
  of the output ids, as the screen), settings.json (the engine's resolved configuration) and ids.json (output ids of
  every request of every rep).
CORRECT=1: untimed correctness pass. 24 fixed questions in the model's chat format with thinking on, each with the JSON
  schema of SP_KW as structured output and no logit_bias, one generate() call each, 384 greedy tokens (EOS allowed to
  stop), with the produced token's logprob -> correct.json ("think_end": whether </think> was produced).
Env: M N IN OL REPS MEMU LLM_KW SP_KW CORRECT."""
import hashlib, json, os, random, statistics, sys, time

THINK_END = 151668
QUESTIONS = [
    "What is the capital of France?", "How many legs does a spider have?", "What is 12 times 7?",
    "Which planet is closest to the sun?", "What colour do you get by mixing blue and yellow?",
    "What is the boiling point of water in Celsius at sea level?", "Who wrote Romeo and Juliet?",
    "What is the square root of 81?", "Name the largest ocean on Earth.", "How many days are in a leap year?",
    "What gas do plants take in for photosynthesis?", "What is the chemical symbol for gold?",
    "How many continents are there?", "What is 100 divided by 4?", "Which animal is known as the king of the jungle?",
    "What is the freezing point of water in Fahrenheit?", "How many sides does a hexagon have?",
    "What language is mainly spoken in Brazil?", "What is the tallest mountain in the world?",
    "What is 15 plus 27?", "Which organ pumps blood through the body?", "How many minutes are in two hours?",
    "What is the main ingredient of guacamole?", "Which metal is liquid at room temperature?",
]


def settings(llm):
    """The resolved engine configuration (simple-valued fields of each sub-config of vllm_config)."""
    import dataclasses, enum
    out = {}
    try:
        vc = llm.llm_engine.vllm_config
        for name in ("model_config", "cache_config", "scheduler_config", "compilation_config", "parallel_config",
                     "load_config", "device_config", "speculative_config", "kv_transfer_config", "observability_config",
                     "structured_outputs_config"):
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
    from vllm.sampling_params import StructuredOutputsParams
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "1")); IN = int(E("IN", "256")); OL = int(E("OL", "512")); REPS = int(E("REPS", "3"))
    kw = {"gpu_memory_utilization": float(E("MEMU", "0.80")), "seed": 0}
    if E("LLM_KW"): kw.update(json.loads(E("LLM_KW")))
    spkw = json.loads(E("SP_KW", "{}"))
    schema = spkw.pop("json_schema", None)
    if "logit_bias" in spkw: spkw["logit_bias"] = {int(k): v for k, v in spkw["logit_bias"].items()}
    if schema is not None: spkw["structured_outputs"] = StructuredOutputsParams(json=schema)

    def promptset(seed):
        r = random.Random(seed)
        return [{"prompt_token_ids": [r.randrange(100, 30000) for _ in range(IN)]} for _ in range(N)]

    t0 = time.perf_counter(); llm = LLM(model=M, **kw); startup = time.perf_counter() - t0
    json.dump(settings(llm), open(os.path.join(out, "settings.json"), "w"), sort_keys=True, default=str)
    if E("CORRECT", "0") == "1":
        tok = llm.get_tokenizer()
        sp = SamplingParams(temperature=0, max_tokens=384, logprobs=0,
                            structured_outputs=StructuredOutputsParams(json=schema) if schema is not None else None)
        outs = []
        for q in QUESTIONS:   # one prompt per call
            text = tok.apply_chat_template([{"role": "user", "content": q + " Reply in JSON with the field answer."}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=True)
            o = llm.generate([text], sp, use_tqdm=False)[0].outputs[0]
            lp = []
            for t, d in zip(o.token_ids, o.logprobs or []):
                lp.append(d[t].logprob if d and t in d else None)
            outs.append({"ids": list(o.token_ids), "lp": lp, "text": o.text, "finish": o.finish_reason,
                         "think_end": THINK_END in o.token_ids})
        json.dump(outs, open(os.path.join(out, "correct.json"), "w"))
        print("CORRECT_DONE", len(outs), "think_end", sum(x["think_end"] for x in outs), flush=True)
        return
    sp = lambda ol: SamplingParams(temperature=0, max_tokens=ol, ignore_eos=True, **spkw)
    t = time.perf_counter(); llm.generate(promptset(1000)[:8], sp(8), use_tqdm=False); warm = time.perf_counter() - t
    ts, ntoks, shas, saved = [], [], [], []
    for rep in range(REPS):
        p = promptset(1 + rep)
        t = time.perf_counter(); res = llm.generate(p, sp(OL), use_tqdm=False); dt = time.perf_counter() - t
        ids = [list(x.outputs[0].token_ids) for x in res]
        ts.append(dt); ntoks.append(sum(len(x) for x in ids)); shas.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16])
        saved.append(ids)
    d = {"startup_s": startup, "warm_s": warm, "per_rep_s": ts, "median_s": statistics.median(ts), "metric_s": statistics.median(ts),
         "out_tokens_per_rep": ntoks, "out_sha_per_rep": shas, "M": M, "N": N, "IN": IN, "OL": OL, "REPS": REPS,
         "LLM_KW": json.loads(json.dumps(kw, default=str)), "SP_KW": E("SP_KW", "")}
    json.dump(saved, open(os.path.join(out, "ids.json"), "w"))
    json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps({k: d[k] for k in ("per_rep_s", "median_s")}), flush=True)


if __name__ == "__main__":   # vLLM may start the engine core with spawn, which re-imports this module
    main()
