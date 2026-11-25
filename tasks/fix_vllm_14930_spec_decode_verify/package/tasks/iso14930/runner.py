"""runner.py <model> : P2 workload driver for ISO-Bench #14930 ([V1][Spec Decode] rejection sampler as Triton kernels).
Engine: LLM(model, speculative_model="[ngram]", num_speculative_tokens=K, ngram_prompt_lookup_max=4,
ngram_prompt_lookup_min=2, max_model_len=4096, gpu_memory_utilization=0.80, seed=0), V1 (VLLM_USE_V1=1), default
compilation / CUDA graphs / prefix caching. Memory utilization is capped at 0.85 whatever the code asks.
Workload (env, defaults = official load):
  N=64 prompts per rep, IN=1024 prompt tokens, OL=256 output tokens (ignore_eos), K=3, TEMP=0 (greedy), REPS=7,
  KIND=lure, DECOY=4: prompt construction (prompt_ids); prompts of the official load come from the curated pool
    (pool14930.py), see prompts_for.
  PRIME=1: before each timed rep an untimed 1-token generate over the same prompts fills the prefix cache, so the timed
    rep is the decode phase (PRIME=0: prefill inside the timed rep; transfer load only).
  SEED_BASE=0: rep r uses pool slice r (warm-up jobs: SEED_BASE=1000, raw seeds outside the pool). Every rep has its
    own prompts, so nothing a run caches about one rep's outputs helps the next rep.
  TEMP>0 (transfer only): per-request seeds (SEED_BASE*100000 + r*1000 + i).
Timing: one small untimed warm-up generate (other prompts), CUDA sync, 2 s idle (the witness separates segments), then
for each rep: [prime] + timed generate, back to back. passes.json: per rep timed wall, prime wall, engine steps and
per-request token increments (from the engine-core outputs, wrapped here in the trusted driver); loop_wall_s = the whole
prime+timed loop (compared with the out-of-process GPU-busy window).
outputs.json: per rep, per request: token ids, text sha256, finish reason, increments (tokens per engine step).
settings.json: effective settings from the frontend config (the engine core's own log is parsed by settings.py)."""
import hashlib, json, os, random, sys
from time import perf_counter, sleep as _sleep, time as _now   # bound before any vllm import
OUT = os.environ["R4_OUT"]; MODEL = sys.argv[1]; E = os.environ.get
N = int(E("N", "64")); IN = int(E("IN", "1024")); OL = int(E("OL", "256")); K = int(E("K", "3")); TEMP = float(E("TEMP", "0"))
REPS = int(E("REPS", "7")); KIND = E("KIND", "lure"); DECOY = int(E("DECOY", "4")); PRIME = int(E("PRIME", "1")); SB = int(E("SEED_BASE", "0"))
POOLSCREEN = int(E("POOLSCREEN", "0"))
LOGPROBS = int(E("LOGPROBS", "0"))
T0 = _now()


def prompt_ids(seed):
    """One prompt: a random 40-token segment (ids 100..29999) repeated up to IN tokens. KIND=lure (official): the FIRST
    copy carries DECOY=4 random substitutions. The n-gram proposer matches the earliest occurrence of the last n tokens,
    i.e. copy 1, so near every decoy it drafts the decoy and the target model (which follows the many clean copies)
    rejects it with a confident prediction: every request gets rejections, at similar rates. KIND=repeat: no decoys."""
    rng = random.Random(seed); seg = [rng.randrange(100, 30000) for _ in range(40)]
    first = list(seg)
    if KIND == "lure":
        for p in rng.sample(range(40), DECOY):
            first[p] = rng.randrange(100, 30000)
    ids = list(first)
    while len(ids) < IN:
        ids += seg
    return ids[:IN]


def pool():
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pool14930 as P
        return list(P.SEL)
    except Exception:
        return None
    finally:
        sys.path.pop(0)


SEL = pool()
def prompts_for(r, n, sb=None):
    """Rep r of the timed load. SEED_BASE=0 with a pool: the curated seeds SEL[r*64 : r*64+n] (pool14930.py, chosen by
    the parent-only pool screen: outputs identical under 3 batch orders, step count near the median). Otherwise raw
    seeds (warm-ups: SEED_BASE=1000 -> seeds 1000*1000 + r*64 + i; never in the pool)."""
    sb = SB if sb is None else sb
    if sb == 0 and SEL and KIND == "lure" and IN == 1024:
        seeds = SEL[r * 64: r * 64 + n]
    else:
        seeds = [sb * 1000 + 500000 + r * 64 + i for i in range(n)]
    return [{"prompt_token_ids": prompt_ids(s)} for s in seeds], seeds


CAP = 0.85
import vllm.config as _vc
_orig_init = _vc.CacheConfig.__init__
def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if self.gpu_memory_utilization > CAP:
        self.gpu_memory_utilization = CAP
_vc.CacheConfig.__init__ = _init
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, speculative_model="[ngram]", num_speculative_tokens=K, ngram_prompt_lookup_max=4,
          ngram_prompt_lookup_min=2, max_model_len=4096, gpu_memory_utilization=0.80, seed=0)
eng = llm.llm_engine


def _g(f):
    try:
        return f()
    except Exception as e:
        return f"error:{type(e).__name__}"


vc = eng.vllm_config if hasattr(eng, "vllm_config") else None
settings = {
    "frontend.spec_k": _g(lambda: vc.speculative_config.num_speculative_tokens),
    "frontend.ngram_min": _g(lambda: vc.speculative_config.ngram_prompt_lookup_min),
    "frontend.ngram_max": _g(lambda: vc.speculative_config.ngram_prompt_lookup_max),
    "frontend.enforce_eager": _g(lambda: vc.model_config.enforce_eager),
    "frontend.dtype": _g(lambda: str(vc.model_config.dtype)),
    "frontend.max_model_len": _g(lambda: vc.model_config.max_model_len),
    "frontend.prefix_caching": _g(lambda: vc.cache_config.enable_prefix_caching),
    "frontend.gpu_memory_utilization": _g(lambda: vc.cache_config.gpu_memory_utilization),
    "frontend.max_num_seqs": _g(lambda: vc.scheduler_config.max_num_seqs),
    "frontend.max_num_batched_tokens": _g(lambda: vc.scheduler_config.max_num_batched_tokens),
    "frontend.compilation_level": _g(lambda: vc.compilation_config.level),
}


json.dump(settings, open(os.path.join(OUT, "settings.json"), "w"), indent=1)

# per-step trace: every engine-core output batch = one engine step; record tokens per request per step
_trace = {"steps": 0, "inc": {}}
_core = eng.engine_core
_orig_get = _core.get_output
def _get_output(*a, **kw):
    o = _orig_get(*a, **kw)
    try:
        outs = o.outputs
        if outs:
            _trace["steps"] += 1
            inc = _trace["inc"]
            for x in outs:
                inc.setdefault(x.request_id, []).append(len(x.new_token_ids))
    except Exception:
        pass
    return o
_core.get_output = _get_output


def sp(ol, r, i=None):
    kw = dict(temperature=TEMP, max_tokens=ol, ignore_eos=True)
    if LOGPROBS:
        kw["logprobs"] = LOGPROBS
    if TEMP > 0 and i is not None:
        kw["seed"] = SB * 100000 + r * 1000 + i
    return SamplingParams(**kw)


# warm-up: other prompts (seed 999 + SEED_BASE), small, untimed
wp, _ = prompts_for(0, 8, sb=999)
llm.generate(wp, [sp(32, 0, i) for i in range(8)], use_tqdm=False)
import torch
torch.cuda.synchronize(); _sleep(2.0)
if POOLSCREEN:
    # parent-only pool screen (operator, once): M candidate seeds, batches of 64 in 3 different orders (different batch
    # partners and positions); per seed: output hash and steps (engine steps the request took) in each order
    M = POOLSCREEN; cand = [10000 + i for i in range(M)]; rnd = random.Random(7)
    orders = {"a": list(cand), "b": cand[32:] + cand[:32], "c": rnd.sample(cand, len(cand))}
    rec = {s: {} for s in cand}
    for on, order in orders.items():
        for j in range(0, M, 64):
            ss = order[j: j + 64]; _trace["inc"] = {}
            res = llm.generate([{"prompt_token_ids": prompt_ids(x)} for x in ss], sp(OL, 0), use_tqdm=False)
            for k, o in enumerate(sorted(res, key=lambda o: int(o.request_id))):
                c = o.outputs[0]; rec[ss[k]][on] = {"h": hashlib.sha256(str(list(c.token_ids)).encode()).hexdigest()[:16],
                                                    "steps": len(_trace["inc"].get(o.request_id, []))}
    json.dump({"OL": OL, "K": K, "IN": IN, "KIND": KIND, "DECOY": DECOY, "rec": {str(k): v for k, v in rec.items()}},
              open(os.path.join(OUT, "pool_screen.json"), "w"))
    sys.exit(0)
reps, outs = [], []
tl0 = perf_counter()
for r in range(REPS):
    P, seeds = prompts_for(r, N)
    tp = None
    if PRIME:
        t = perf_counter(); llm.generate(P, sp(1, r), use_tqdm=False); tp = perf_counter() - t
    _trace["steps"] = 0; _trace["inc"] = {}
    params = [sp(OL, r, i) for i in range(N)]
    t = perf_counter(); res = llm.generate(P, params, use_tqdm=False); dt = perf_counter() - t
    steps = _trace["steps"]; inc = dict(_trace["inc"])
    po = []
    for o in sorted(res, key=lambda o: int(o.request_id)):
        c = o.outputs[0]
        d = {"ids": list(c.token_ids), "text_sha256": hashlib.sha256(c.text.encode()).hexdigest(), "finish": c.finish_reason,
             "inc": inc.get(o.request_id, [])}
        if LOGPROBS:
            d["top"] = [{str(t): float(v.logprob) for t, v in s.items()} for s in (c.logprobs or [])]
        po.append(d)
    hist = {}
    for d in po:
        for x in d["inc"]:
            hist[str(x)] = hist.get(str(x), 0) + 1
    rec = {"rep": r, "wall_s": dt, "prime_s": tp, "steps": steps, "out_tokens": sum(len(d["ids"]) for d in po),
           "inc_hist": hist, "t_end_unix": _now()}
    reps.append(rec); outs.append(po); print("[P2REP]", json.dumps(rec), flush=True)
loop = perf_counter() - tl0
json.dump({"workload": {"N": N, "IN": IN, "OL": OL, "K": K, "TEMP": TEMP, "KIND": KIND, "DECOY": DECOY, "POOL": bool(SEL), "PRIME": PRIME, "SEED_BASE": SB,
                        "REPS": REPS}, "reps": outs}, open(os.path.join(OUT, "outputs.json"), "w"))
json.dump({"reps": reps, "loop_wall_s": loop, "process_wall_s": _now() - T0,
           "workload": {"N": N, "IN": IN, "OL": OL, "K": K, "TEMP": TEMP, "KIND": KIND, "DECOY": DECOY, "POOL": bool(SEL), "PRIME": PRIME, "SEED_BASE": SB,
                        "REPS": REPS}}, open(os.path.join(OUT, "passes.json"), "w"), indent=1)
