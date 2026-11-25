"""runner.py <model> <timed|correct> : P2 workload driver for ISO-Bench #13365 ([V1][Spec Decode] Optimize N-gram
matching with Numba). Trusted measurement code (not in the candidate's overlay); written 2026-11-24 for the iso13365 card.

Engine (the PR's setting, scaled to one RTX 4090): V1 (VLLM_USE_V1=1), LLM(model, speculative_model="[ngram]",
num_speculative_tokens=5, ngram_prompt_lookup_max=4, ngram_prompt_lookup_min=3, max_model_len=4096,
gpu_memory_utilization=0.80, seed=0); defaults otherwise (torch.compile + CUDA graphs, prefix caching, chunked prefill,
engine core IN-PROCESS (VLLM_ENABLE_V1_MULTIPROCESSING=0, see below). At this commit the model runner calls NgramProposer.propose(context, n=min=3, k=5)
for every request after every step, serially before the next step is launched.

timed (official load, env N=32 IN=2048 OL=128 KIND=repeat SEG=40 REPS=6 SEED_BASE=0):
  prompt set r: N prompts of IN token ids; KIND=repeat: each prompt is its own random SEG-token segment repeated
  (KIND=random: random ids, no repeats; KIND=late: IN/2 random ids, then a repeated segment).
  untimed warm-up: one full rep on prompt set SEED_BASE+10000 (also triggers any lazy JIT of the proposer);
  then 2 s idle (the out-of-process witness needs a gap), then for r in 0..REPS-1:
    prime  (untimed): generate(set r, max_tokens=1)  -> the prompts' KV goes into the prefix cache;
    timed          : generate(set r, max_tokens=OL, greedy, ignore_eos) -> decode-dominated, every step runs the proposer.
  Each rep uses NEW prompts, so nothing a candidate memoises in one rep can be reused in the next.
  metric = sum of the REPS timed walls (s).
correct: five fixed prompt groups (v2 adds 'far': the only match lies 300-3000 tokens back), 48 greedy tokens each, plus a direct unit test of NgramProposer.propose on 400
  fixed contexts.
Why in-process (smoke 2026-11-24 08:15 UTC): with the default multiprocessing engine core, generate() adds its requests
one by one while the core is already stepping; with prefix-cache hits each request starts decoding on arrival, so one
rep's 32 requests ran in staggered waves (31 to 122 engine steps for the same kind of rep, timed reps 0.42-1.03 s) and
batch composition, step counts and counters differed from run to run. In-process, all requests are queued before the
first step: batches, outputs and counters are reproducible. The proposer still runs serially between steps.

Counters (read-only wrappers on LOCKED classes, installed before the engine starts):
  core (vllm.v1.core.scheduler.Scheduler.update_from_output): per generate() call, steps,
    request-steps, scheduled spec tokens, accepted spec tokens, proposed tokens, non-empty proposals, longest proposal,
    histogram of tokens produced per request-step, and a hash of every proposal (request id + token ids, every step).
    Dumped as one line of core_counters.jsonl when the scheduler becomes idle (= the end of each generate() call).
  front end (OutputProcessor.process_outputs): engine steps seen, outputs, tokens, per generate() call.
  config (LLMEngine.__init__): the VllmConfig the engine was built with -> settings.json.
Writes to $R4_OUT: outputs.json, passes.json, settings.json, core_counters.jsonl (correct mode: correct.json)."""
import hashlib, json, os, random, sys
from time import perf_counter, sleep as _sleep, time as _now   # bound before any vllm import
OUT = os.environ["R4_OUT"]; MODEL = sys.argv[1]; MODE = sys.argv[2] if len(sys.argv) > 2 else "timed"
E = os.environ.get
N = int(E("N", "32")); IN = int(E("IN", "2048")); OL = int(E("OL", "128")); KIND = E("KIND", "repeat")
SEG = int(E("SEG", "40")); REPS = int(E("REPS", "6")); SB = int(E("SEED_BASE", "0"))
K, NMIN, NMAX = 5, 3, 4
T0 = _now()
CORE_FILE = os.path.join(OUT, "core_counters.jsonl")


def prompts_for(seed, n=N, inlen=IN, kind=KIND, seg=SEG):
    rng = random.Random(1_000_003 * (seed + 1) + 13365)
    tok = lambda m: [rng.randrange(100, 30000) for _ in range(m)]
    P = []
    for _ in range(n):
        if kind == "repeat":
            s = tok(seg); P.append((s * (inlen // seg + 1))[:inlen])
        elif kind == "random":
            P.append(tok(inlen))
        elif kind == "late":
            h = inlen // 2; s = tok(seg); P.append(tok(h) + (s * ((inlen - h) // seg + 1))[: inlen - h])
        elif kind == "smallvocab":   # correct mode only: short prompts over 4 token ids, so n-gram matches are everywhere
            P.append([100 + rng.randrange(4) for _ in range(inlen)])
        elif kind == "far":   # correct mode only (v2): the prompt ends with the first 3 tokens of a segment that occurs only
            # once, far back (300-3000 tokens), so the first proposal must come from far back; a capped search proposes nothing
            s = tok(seg); p = [0, 500, 1000, 1500, 2000, 2400, 2600, 2700][len(P) % 8]
            body = tok(inlen - 3); body[p:p + seg] = s; P.append(body[: inlen - 3] + s[:3])
        else:
            raise SystemExit(f"bad KIND {kind}")
    return P


# ---------------- counters ----------------
_core = {}
def _core_reset():
    _core.clear()
    _core.update(steps=0, req_steps=0, spec_req_steps=0, scheduled_spec=0, accepted=0, proposed=0,
                 proposals_nonempty=0, max_proposal=0, gen_hist={}, h=hashlib.sha256())
_core_reset()
def _core_dump():
    rec = {k: v for k, v in _core.items() if k != "h"}
    rec["prop_hash"] = _core["h"].hexdigest()[:20]; rec["pid"] = os.getpid()
    with open(CORE_FILE, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    _core_reset()
hooks = []
import vllm.v1.core.scheduler as _S
_orig_uf = _S.Scheduler.update_from_output
def _update_from_output(self, so, mro):
    try:
        nst = so.num_scheduled_tokens
        if nst:
            st = _core; spec = so.scheduled_spec_decode_tokens or {}
            sampled = mro.sampled_token_ids; props = mro.spec_token_ids; idx = mro.req_id_to_index
            st["steps"] += 1; items = []
            for rid in sorted(nst):
                i = idx.get(rid)
                if i is None:
                    continue
                st["req_steps"] += 1
                g = len(sampled[i]); key = str(g); st["gen_hist"][key] = st["gen_hist"].get(key, 0) + 1
                if rid in spec:
                    st["spec_req_steps"] += 1; st["scheduled_spec"] += len(spec[rid]); st["accepted"] += max(0, g - 1)
                p = props[i] if (props is not None and i < len(props)) else None
                p = [int(x) for x in p] if p is not None and len(p) else []
                if p:
                    st["proposed"] += len(p); st["proposals_nonempty"] += 1
                    st["max_proposal"] = max(st["max_proposal"], len(p))
                items.append((rid, p))
            st["h"].update(repr(items).encode())
    except Exception as e:   # recorded, so a broken counter is visible (it makes the check fail, never pass)
        _core["error"] = repr(e)[:200]
    r = _orig_uf(self, so, mro)
    if not self.running and not self.waiting:
        _core_dump()
    return r
_S.Scheduler.update_from_output = _update_from_output; hooks.append("core")
_fe = {}
def _fe_reset():
    _fe.clear(); _fe.update(steps=0, outputs=0, tokens=0)
_fe_reset()
import vllm.v1.engine.output_processor as _OPm
_orig_po = _OPm.OutputProcessor.process_outputs
def _process_outputs(self, engine_core_outputs, *a, **kw):
    if engine_core_outputs:
        _fe["steps"] += 1; _fe["outputs"] += len(engine_core_outputs)
        _fe["tokens"] += sum(len(o.new_token_ids) for o in engine_core_outputs)
    return _orig_po(self, engine_core_outputs, *a, **kw)
_OPm.OutputProcessor.process_outputs = _process_outputs; hooks.append("frontend")
_cfg = {}
import vllm.v1.engine.llm_engine as _LEm
_orig_li = _LEm.LLMEngine.__init__
def _llm_init(self, *a, **kw):
    _cfg["vc"] = kw.get("vllm_config", a[0] if a else None)
    return _orig_li(self, *a, **kw)
_LEm.LLMEngine.__init__ = _llm_init; hooks.append("config")

from vllm import LLM, SamplingParams
import vllm.envs as _envs
llm = LLM(model=MODEL, speculative_model="[ngram]", num_speculative_tokens=K, ngram_prompt_lookup_max=NMAX,
          ngram_prompt_lookup_min=NMIN, max_model_len=4096, gpu_memory_utilization=0.80, seed=0)


def _g(f):
    try:
        v = f()
        return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)
    except Exception as e:
        return f"error:{type(e).__name__}"
vc = _cfg.get("vc")
settings = {
    "spec_num_speculative_tokens": _g(lambda: vc.speculative_config.num_speculative_tokens),
    "spec_ngram_min": _g(lambda: vc.speculative_config.ngram_prompt_lookup_min),
    "spec_ngram_max": _g(lambda: vc.speculative_config.ngram_prompt_lookup_max),
    "spec_draft_model": _g(lambda: vc.speculative_config.draft_model_config.model),
    "model": _g(lambda: vc.model_config.model), "dtype": _g(lambda: str(vc.model_config.dtype)),
    "max_model_len": _g(lambda: vc.model_config.max_model_len), "enforce_eager": _g(lambda: vc.model_config.enforce_eager),
    "seed": _g(lambda: vc.model_config.seed),
    "block_size": _g(lambda: vc.cache_config.block_size), "enable_prefix_caching": _g(lambda: vc.cache_config.enable_prefix_caching),
    "gpu_memory_utilization": _g(lambda: vc.cache_config.gpu_memory_utilization),
    "max_num_seqs": _g(lambda: vc.scheduler_config.max_num_seqs),
    "max_num_batched_tokens": _g(lambda: vc.scheduler_config.max_num_batched_tokens),
    "chunked_prefill": _g(lambda: vc.scheduler_config.chunked_prefill_enabled),
    "compilation_level": _g(lambda: vc.compilation_config.level),
    "use_cudagraph": _g(lambda: vc.compilation_config.use_cudagraph),
    "n_cudagraph_capture_sizes": _g(lambda: len(vc.compilation_config.cudagraph_capture_sizes)),
    "vllm_use_v1": _g(lambda: bool(_envs.VLLM_USE_V1)),
    "v1_multiprocessing": _g(lambda: bool(_envs.VLLM_ENABLE_V1_MULTIPROCESSING)),
    "engine_core_client": _g(lambda: type(llm.llm_engine.engine_core).__name__),
    "hooks": ",".join(hooks),
}
# live engine core (in-process): what the model runner really uses
live = {}
try:
    _core_obj = llm.llm_engine.engine_core.engine_core
    _w = getattr(_core_obj.model_executor, "driver_worker", None); _w = getattr(_w, "worker", _w); mr = _w.model_runner
    live = {"use_spec_decode": _g(lambda: mr.use_spec_decode), "drafter": _g(lambda: type(mr.drafter).__name__),
            "k": _g(lambda: mr.speculative_config.num_speculative_tokens),
            "nmin": _g(lambda: mr.speculative_config.ngram_prompt_lookup_min),
            "nmax": _g(lambda: mr.speculative_config.ngram_prompt_lookup_max),
            "use_cuda_graph": _g(lambda: mr.use_cuda_graph), "n_cudagraph_sizes": _g(lambda: len(mr.cudagraph_batch_sizes)),
            "max_num_reqs": _g(lambda: mr.max_num_reqs), "scheduler": _g(lambda: type(_core_obj.scheduler).__name__),
            "num_gpu_blocks": _g(lambda: _core_obj.scheduler.cache_config.num_gpu_blocks)}
except Exception as e:
    live = {"error": repr(e)[:200]}
settings.update({"live_" + k: v for k, v in live.items() if k != "num_gpu_blocks"})
greedy = lambda ol: SamplingParams(temperature=0, max_tokens=ol, ignore_eos=True)


def gen(prompts, ol):
    _fe_reset()
    t = perf_counter(); res = llm.generate([{"prompt_token_ids": p} for p in prompts], greedy(ol), use_tqdm=False)
    dt = perf_counter() - t
    outs = []
    for o in sorted(res, key=lambda o: int(o.request_id)):
        c = o.outputs[0]
        outs.append({"ids": list(c.token_ids), "sha": hashlib.sha256(c.text.encode()).hexdigest()[:16], "fin": c.finish_reason})
    return dt, dict(_fe), outs


def core_lines():
    try:
        return [json.loads(l) for l in open(CORE_FILE)]
    except Exception:
        return []


if MODE == "correct":
    groups = {"repeat": prompts_for(900, 8, 1024, "repeat"), "random": prompts_for(901, 8, 1024, "random"),
              "late": prompts_for(902, 8, 1024, "late"), "smallvocab": prompts_for(903, 8, 24, "smallvocab"), "far": prompts_for(904, 8, 3000, "far")}
    res = {"groups": {}, "settings": settings, "live": live}
    n0 = len(core_lines())
    for name, P in groups.items():
        dt, fe, outs = gen(P, 48)
        res["groups"][name] = {"ids": [o["ids"] for o in outs], "sha": [o["sha"] for o in outs], "fe": fe, "s": round(dt, 3)}
    cl = core_lines()[n0:]
    for name, c in zip(groups, cl):
        res["groups"][name]["core"] = c
    res["core_lines"] = len(cl)
    # direct unit test of the proposer's contract (first match of the last n tokens, the k tokens after it)
    import numpy as np
    unit = {"api": "ok", "results": []}
    try:
        from vllm.v1.spec_decode.ngram_proposer import NgramProposer
        prop = NgramProposer()
        rs = np.random.RandomState(13365)
        cases = []
        for t in range(360):
            V = [2, 3, 4, 8, 50, 30000][t % 6]; L = int(rs.choice([1, 2, 3, 4, 5, 8, 17, 64, 300, 2000, 4000]))
            cases.append((rs.randint(100, 100 + V, size=L).astype(np.int32), int(rs.randint(1, 6)), int(rs.randint(1, 9))))
        for t in range(40):   # structured: periodic, pattern twice with different continuations, match only near the end
            L = 50 + 37 * t; base = rs.randint(100, 30000, size=L).astype(np.int32)
            if t % 3 == 0:
                seg = rs.randint(100, 30000, size=7 + t % 11).astype(np.int32); ctx = np.resize(seg, L)
            elif t % 3 == 1:
                ctx = base.copy(); ctx[-3:] = ctx[5:8]; ctx[L // 2:L // 2 + 3] = ctx[5:8]
            else:
                ctx = base.copy(); ctx[-3:] = ctx[L - 20:L - 17]
            cases.append((ctx.astype(np.int32), 3, 5))
        t0 = perf_counter()
        for ctx, n, k in cases:
            try:
                r = prop.propose(ctx, n, k)
                unit["results"].append(None if r is None else [int(x) for x in r])
            except Exception as e:
                unit["results"].append(f"error:{type(e).__name__}")
        unit["s"] = round(perf_counter() - t0, 4); unit["n"] = len(cases)
    except Exception as e:
        unit = {"api": f"error:{type(e).__name__}:{str(e)[:120]}", "results": []}
    res["unit"] = unit
    json.dump(res, open(os.path.join(OUT, "correct.json"), "w"))
    json.dump(settings, open(os.path.join(OUT, "settings.json"), "w"), indent=1)
    print("[P2] correct done", json.dumps({k: v.get("core") for k, v in res["groups"].items()})[:600], flush=True)
    sys.exit(0)

# ---------------- timed ----------------
warm_s = []
W = prompts_for(SB + 10_000)
for ol in (1, OL):
    dt, _, _ = gen(W, ol); warm_s.append(dt)
_sleep(2.0)   # GPU idle before the measured region: the out-of-process witness takes the longest busy segment
prime_s, timed_s, fe_timed, fe_prime, reps, primes = [], [], [], [], [], []
rt0 = perf_counter()
for r in range(REPS):
    P = prompts_for(SB + r)
    dt, fe, o = gen(P, 1); prime_s.append(dt); fe_prime.append(fe); primes.append(o)
    dt, fe, o = gen(P, OL); timed_s.append(dt); fe_timed.append(fe); reps.append(o)
region_s = perf_counter() - rt0
cl = core_lines()
key = {"N": N, "IN": IN, "OL": OL, "KIND": KIND, "SEG": SEG, "REPS": REPS, "SEED_BASE": SB}
json.dump({"key": key, "passes": reps, "primes": primes, "core": cl, "fe_timed": fe_timed, "fe_prime": fe_prime, "live": live},
          open(os.path.join(OUT, "outputs.json"), "w"))
json.dump(settings | {"max_proposal_seen": max([c.get("max_proposal", 0) for c in cl] or [0])},
          open(os.path.join(OUT, "settings.json"), "w"), indent=1)
json.dump({"timed_s": timed_s, "prime_s": prime_s, "warm_s": warm_s, "region_s": region_s, "fe_timed": fe_timed,
           "core_lines": len(cl), "core_pids": sorted({c.get("pid") for c in cl}), "runner_pid": os.getpid(),
           "process_wall_s": _now() - T0, "key": key}, open(os.path.join(OUT, "passes.json"), "w"), indent=1)
print("[P2] timed", json.dumps({"timed_s": [round(x, 4) for x in timed_s], "sum": round(sum(timed_s), 4),
                                "prime_s": [round(x, 3) for x in prime_s], "core_lines": len(cl)}), flush=True)
